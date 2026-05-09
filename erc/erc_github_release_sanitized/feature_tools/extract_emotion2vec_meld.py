"""Extract MELD audio features with emotion2vec. Paths are passed through CLI."""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

LABEL_MAP = {"neutral":0,"joy":1,"surprise":2,"anger":3,"sadness":4,"disgust":5,"fear":6}
MAX_FRAMES = 300


def load_model(emotion2vec_repo, checkpoint_path, gpu):
    sys.path.insert(0, emotion2vec_repo)
    import dataclasses
    import fairseq
    import yaml
    from dataclasses import dataclass
    from omegaconf import OmegaConf

    @dataclass
    class UserDirModule:
        user_dir: str

    fairseq.utils.import_user_module(UserDirModule(os.path.join(emotion2vec_repo, "upstream")))
    from upstream.models.emotion2vec import Data2VecMultiConfig, Data2VecMultiModel

    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(Data2VecMultiConfig)}
    conf = {k: v for k, v in raw_cfg["model_conf"].items() if k in valid}
    if "modalities" in conf:
        for k in list(conf["modalities"].keys()):
            if k != "audio": conf["modalities"].pop(k, None)
    model_cfg = OmegaConf.structured(Data2VecMultiConfig)
    OmegaConf.set_struct(model_cfg, False)
    model_cfg = OmegaConf.merge(model_cfg, OmegaConf.create(conf))
    model = Data2VecMultiModel(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = {k[len("d2v_model."):]: v for k, v in ckpt["model"].items() if k.startswith("d2v_model.")}
    model.load_state_dict(state, strict=False)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    normalize = ckpt["cfg"]["task"].get("normalize", True)
    return model, normalize, device


def extract_one(model, normalize, wav_path, device):
    wav, sr = sf.read(wav_path)
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz wav, got {sr}: {wav_path}")
    if len(wav.shape) > 1:
        wav = wav[:, 0]
    with torch.no_grad():
        source = torch.from_numpy(wav).float().to(device)
        if normalize:
            source = F.layer_norm(source, source.shape)
        res = model.extract_features(source.view(1, -1), padding_mask=None, remove_extra_tokens=True)
        feats = res["x"].squeeze(0)
    if feats.shape[0] >= MAX_FRAMES:
        return feats[:MAX_FRAMES].cpu()
    pad = torch.zeros(MAX_FRAMES - feats.shape[0], feats.shape[1], device=feats.device)
    return torch.cat([feats, pad], dim=0).cpu()


def parse_csv(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            emo = r["Emotion"].strip().lower()
            if emo not in LABEL_MAP: continue
            did, uid = int(r["Dialogue_ID"]), int(r["Utterance_ID"])
            rows.append({"id_str": f"dia{did}_utt{uid}", "label": LABEL_MAP[emo], "dialogue_id": did, "utterance_id": uid})
    return rows


def process_split(split, meld_root, output_dir, model, normalize, device, out_split=None):
    raw_split = "dev" if split == "val" else split
    out_split = out_split or ("val" if raw_split == "dev" else raw_split)
    wav_dir = os.path.join(meld_root, raw_split, f"{raw_split}_splits")
    csv_path = os.path.join(meld_root, raw_split, f"{raw_split}_sent_emo.csv")
    results = []
    for item in tqdm(parse_csv(csv_path), desc=out_split):
        wav_path = os.path.join(wav_dir, f"{item['id_str']}.wav")
        if os.path.exists(wav_path):
            try:
                audio = extract_one(model, normalize, wav_path, device)
            except Exception:
                audio = torch.zeros(MAX_FRAMES, 1024)
        else:
            audio = torch.zeros(MAX_FRAMES, 1024)
        results.append({**item, "audio": audio})
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"audio_{out_split}.pkl"), "wb") as f:
        pickle.dump(results, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotion2vec_repo", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--meld_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", default="all", choices=["train","val","dev","test","all"],
                    help="Use train/val/test for exported files. MELD raw validation split is dev; val maps to dev.")
    args = ap.parse_args()
    model, normalize, device = load_model(args.emotion2vec_repo, args.checkpoint_path, args.gpu)
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, args.meld_root, args.output_dir, model, normalize, device)


if __name__ == "__main__":
    main()
