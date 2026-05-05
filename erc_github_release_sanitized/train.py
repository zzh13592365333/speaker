import argparse
import json
import os
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from erc.datasets import DatasetConfig, MultimodalDataset, mm_collate_fn
from erc.loss import InfoNCE, SupConLoss
from erc.model import MultimodalERCModel, UnimodalERCModel
from erc.utils import LABELS, compute_metrics, set_seed


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(msg: str, log_path: str = None):
    print(msg, flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def evaluate(model, loader, device, dataset=None, collect_details=False):
    model.eval()
    preds, labels, sample_idx = [], [], []
    gate_values, visual_confs, aux_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            out = model(batch)
            p = torch.argmax(out["logits"], dim=1).cpu().tolist()
            y = batch["labels"].cpu().tolist()
            idx = batch["sample_idx"].cpu().tolist()
            preds.extend(p); labels.extend(y); sample_idx.extend(idx)
            if collect_details:
                if "v_logits_aux" in out:
                    probs = torch.softmax(out["v_logits_aux"], dim=-1)
                    visual_confs.extend(torch.max(probs, dim=-1).values.cpu().tolist())
                    aux_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
                else:
                    visual_confs.extend([None] * len(p)); aux_preds.extend([None] * len(p))
                if "gate_value" in out:
                    gate_values.extend(out["gate_value"].view(-1).detach().cpu().tolist())
                else:
                    gate_values.extend([None] * len(p))
    ids = [dataset.ids_arr[i] for i in sample_idx] if dataset is not None else None
    details = None
    if collect_details:
        details = {"gate_values": gate_values, "visual_confidences": visual_confs, "aux_preds": aux_preds}
    return compute_metrics(preds, labels), labels, preds, ids, details


def save_predictions(ids, labels, preds, label_names, path):
    data = {}
    for uid, gt, pred in zip(ids, labels, preds):
        data[str(uid)] = {
            "gt": int(gt), "pred": int(pred),
            "gt_label": label_names[int(gt)], "pred_label": label_names[int(pred)],
            "correct": bool(int(gt) == int(pred)),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_tag(args):
    parts = [args.dataset, args.modalities.replace(",", "_"), args.context_mode, f"seed{args.seed}"]
    for flag in ["disable_context", "disable_crossmodal", "disable_audio_video_interaction", "disable_text_video_interaction", "disable_delta_diff", "disable_cl"]:
        if getattr(args, flag, False):
            parts.append(flag.replace("disable_", "no"))
    if not args.use_conf_gate:
        parts.append("nogate")
    return "_".join(parts)


def train(args):
    cfg_dict = load_config(args.config)
    cfg = DatasetConfig.from_dict(cfg_dict)
    dataset_name = args.dataset or cfg.dataset
    label_names = cfg_dict.get("label_names", LABELS[dataset_name])
    args.num_classes = len(label_names)
    class_weights = cfg_dict.get("class_weights", [1.0] * args.num_classes)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    modes: List[str] = [m.strip() for m in args.modalities.split(",") if m.strip()]
    args.modalities = ",".join(modes)

    tag = build_tag(args)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(args.log_dir, f"train_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    open(log_path, "w", encoding="utf-8").close()

    log(f"Dataset: {dataset_name}", log_path)
    log(f"Modalities: {modes}", log_path)
    log(f"Device: {device}", log_path)
    log(json.dumps(vars(args), indent=2, ensure_ascii=False), log_path)

    train_set = MultimodalDataset("train", modes, cfg, args.window_size, training=True, context_mode=args.context_mode)
    val_set = MultimodalDataset("val", modes, cfg, args.window_size, training=False, context_mode=args.context_mode)
    test_set = MultimodalDataset("test", modes, cfg, args.window_size, training=False, context_mode=args.context_mode)
    log(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", log_path)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=mm_collate_fn, num_workers=args.num_workers, generator=generator, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=mm_collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=mm_collate_fn, num_workers=args.num_workers, pin_memory=True)

    if len(modes) == 1:
        args.modality_type = modes[0]
        model = UnimodalERCModel(args).to(device)
    else:
        model = MultimodalERCModel(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=args.label_smoothing)
    criterion_infonce = InfoNCE(args.infonce_temp).to(device)
    criterion_supcon = SupConLoss(args.supcon_temp).to(device)

    best_val_f1, best_test_f1 = 0.0, 0.0
    best_val_report, best_test_report = "", ""

    for epoch in range(args.epochs):
        model.train()
        cl_weight = args.cl_weight * (args.cl_decay ** epoch)
        sums = {"loss": 0.0, "cls": 0.0, "aux": 0.0, "cl": 0.0}
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss_cls = criterion_cls(out["logits"], batch["labels"])
            loss_cl = torch.tensor(0.0, device=device)
            if not args.disable_cl and "cl_features" in out:
                cl = out["cl_features"]
                for a, b in [("t", "a"), ("t", "v"), ("a", "v")]:
                    if a in cl and b in cl:
                        loss_cl = loss_cl + criterion_infonce(cl[a], cl[b])
                if "fused" in cl:
                    loss_cl = loss_cl + criterion_supcon(cl["fused"], batch["labels"])
            loss_aux = torch.tensor(0.0, device=device)
            if "v_logits_aux" in out and args.aux_loss_weight > 0:
                loss_aux = criterion_cls(out["v_logits_aux"], batch["labels"])
            loss = loss_cls + cl_weight * loss_cl + args.aux_loss_weight * loss_aux
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            steps += 1
            sums["loss"] += float(loss.item()); sums["cls"] += float(loss_cls.item()); sums["aux"] += float(loss_aux.item()); sums["cl"] += float(loss_cl.item())
            pbar.set_postfix({k: f"{v / steps:.3f}" for k, v in sums.items()})
        scheduler.step()
        log(f"[Epoch {epoch+1}] train " + " ".join([f"{k}={v/max(1,steps):.4f}" for k, v in sums.items()]), log_path)

        val_metrics, v_lab, v_pred, v_ids, _ = evaluate(model, val_loader, device, val_set)
        test_metrics, t_lab, t_pred, t_ids, _ = evaluate(model, test_loader, device, test_set, collect_details=True)
        val_report = classification_report(v_lab, v_pred, target_names=label_names, digits=4, zero_division=0)
        test_report = classification_report(t_lab, t_pred, target_names=label_names, digits=4, zero_division=0)
        log(f"Epoch {epoch+1}: val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f}", log_path)
        log(test_report, log_path)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_report = test_report
            torch.save(model.state_dict(), f"best_val_model_{tag}.pth")
            save_predictions(t_ids, t_lab, t_pred, label_names, f"pred_{tag}_best_val_test.json")
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            best_test_report = test_report
            torch.save(model.state_dict(), f"best_test_model_{tag}.pth")
            save_predictions(t_ids, t_lab, t_pred, label_names, f"pred_{tag}_best_test.json")

    log("\nFinal summary", log_path)
    log(f"Best val-selected test report:\n{best_val_report}", log_path)
    log(f"Best test report:\n{best_test_report}", log_path)
    return best_test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config JSON")
    parser.add_argument("--dataset", default="", choices=["", "meld", "iemocap"])
    parser.add_argument("--modalities", default="text,audio,video")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--context_mode", default="past", choices=["past", "full", "center"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_dim", type=int, default=1024)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--video_dim", type=int, default=4096)
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--disable_crossmodal", action="store_true")
    parser.add_argument("--disable_audio_video_interaction", action="store_true")
    parser.add_argument("--disable_text_video_interaction", action="store_true")
    parser.add_argument("--disable_delta_diff", action="store_true")
    parser.add_argument("--use_conf_gate", action="store_true", default=True)
    parser.add_argument("--no_conf_gate", action="store_false", dest="use_conf_gate")
    parser.add_argument("--aux_loss_weight", type=float, default=0.15)
    parser.add_argument("--disable_cl", action="store_true")
    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--cl_decay", type=float, default=0.95)
    parser.add_argument("--infonce_temp", type=float, default=0.1)
    parser.add_argument("--supcon_temp", type=float, default=0.07)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--log_file", default="")
    args = parser.parse_args()
    train(args)
