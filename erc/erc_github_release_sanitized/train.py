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
from erc.model import MultimodalERCModel
from erc.utils import LABELS, compute_metrics, set_seed


FULL_MODALITIES: List[str] = ["text", "audio", "video"]
CONTEXT_MODE = "past"


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
            preds.extend(p)
            labels.extend(y)
            sample_idx.extend(idx)

            if collect_details:
                probs = torch.softmax(out["v_logits_aux"], dim=-1)
                visual_confs.extend(torch.max(probs, dim=-1).values.cpu().tolist())
                aux_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
                gate_values.extend(out["gate_value"].view(-1).detach().cpu().tolist())

    ids = [dataset.ids_arr[i] for i in sample_idx] if dataset is not None else None
    details = None
    if collect_details:
        details = {
            "gate_values": gate_values,
            "visual_confidences": visual_confs,
            "aux_preds": aux_preds,
        }
    return compute_metrics(preds, labels), labels, preds, ids, details


def save_predictions(ids, labels, preds, label_names, path):
    data = {}
    for uid, gt, pred in zip(ids, labels, preds):
        data[str(uid)] = {
            "gt": int(gt),
            "pred": int(pred),
            "gt_label": label_names[int(gt)],
            "pred_label": label_names[int(pred)],
            "correct": bool(int(gt) == int(pred)),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_tag(dataset: str, seed: int) -> str:
    return f"{dataset}_text_audio_video_past_seed{seed}"


def train(args):
    cfg_dict = load_config(args.config)
    cfg = DatasetConfig.from_dict(cfg_dict)
    dataset_name = args.dataset or cfg.dataset
    label_names = cfg_dict.get("label_names", LABELS[dataset_name])
    args.num_classes = len(label_names)
    class_weights = cfg_dict.get("class_weights", [1.0] * args.num_classes)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    modes = FULL_MODALITIES
    tag = build_tag(dataset_name, args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = args.log_file or os.path.join(
        args.log_dir,
        f"train_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    open(log_path, "w", encoding="utf-8").close()

    log(f"Dataset: {dataset_name}", log_path)
    log(f"Modalities: {modes}", log_path)
    log(f"Context mode: {CONTEXT_MODE}", log_path)
    log(f"Device: {device}", log_path)
    log(json.dumps(vars(args), indent=2, ensure_ascii=False), log_path)

    train_set = MultimodalDataset("train", modes, cfg, args.window_size, training=True, context_mode=CONTEXT_MODE)
    val_set = MultimodalDataset("val", modes, cfg, args.window_size, training=False, context_mode=CONTEXT_MODE)
    test_set = MultimodalDataset("test", modes, cfg, args.window_size, training=False, context_mode=CONTEXT_MODE)
    log(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", log_path)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=mm_collate_fn,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mm_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mm_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MultimodalERCModel(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion_cls = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    criterion_infonce = InfoNCE(args.infonce_temp).to(device)
    criterion_supcon = SupConLoss(args.supcon_temp).to(device)

    best_val_f1 = -1.0
    best_val_test_f1 = 0.0
    best_val_report = ""

    for epoch in range(args.epochs):
        model.train()
        cl_weight = args.cl_weight * (args.cl_decay ** epoch)
        sums = {"loss": 0.0, "cls": 0.0, "aux": 0.0, "cl": 0.0}
        steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)

        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss_cls = criterion_cls(out["logits"], batch["labels"])

            cl = out["cl_features"]
            loss_cl = torch.tensor(0.0, device=device)
            for a, b in [("t", "a"), ("t", "v"), ("a", "v")]:
                loss_cl = loss_cl + criterion_infonce(cl[a], cl[b])
            loss_cl = loss_cl + criterion_supcon(cl["fused"], batch["labels"])

            loss_aux = criterion_cls(out["v_logits_aux"], batch["labels"])
            loss = loss_cls + cl_weight * loss_cl + args.aux_loss_weight * loss_aux
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            steps += 1
            sums["loss"] += float(loss.item())
            sums["cls"] += float(loss_cls.item())
            sums["aux"] += float(loss_aux.item())
            sums["cl"] += float(loss_cl.item())
            pbar.set_postfix({k: f"{v / steps:.3f}" for k, v in sums.items()})

        scheduler.step()
        log(
            f"[Epoch {epoch + 1}] train "
            + " ".join([f"{k}={v / max(1, steps):.4f}" for k, v in sums.items()]),
            log_path,
        )

        val_metrics, v_lab, v_pred, _, _ = evaluate(model, val_loader, device, val_set)
        test_metrics, t_lab, t_pred, t_ids, _ = evaluate(model, test_loader, device, test_set, collect_details=True)
        log(
            f"Epoch {epoch + 1}: "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f}",
            log_path,
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_test_f1 = test_metrics["f1"]
            best_val_report = classification_report(t_lab, t_pred, target_names=label_names, digits=4, zero_division=0)
            torch.save(model.state_dict(), f"best_val_model_{tag}.pth")
            save_predictions(t_ids, t_lab, t_pred, label_names, f"pred_{tag}_best_val_test.json")
            log(
                f"New best val checkpoint: val_f1={best_val_f1:.4f}; "
                f"corresponding test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f}",
                log_path,
            )
            log(best_val_report, log_path)

    log("\nFinal summary", log_path)
    log(f"Best val F1: {best_val_f1:.4f}; corresponding test F1: {best_val_test_f1:.4f}", log_path)
    log(f"Best val-selected test report:\n{best_val_report}", log_path)
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to dataset config JSON")
    parser.add_argument("--dataset", default="", choices=["", "meld", "iemocap"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_dim", type=int, default=1024)
    parser.add_argument("--audio_dim", type=int, default=1024)
    parser.add_argument("--video_dim", type=int, default=4096)
    parser.add_argument("--aux_loss_weight", type=float, default=0.15)
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
