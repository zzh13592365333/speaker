#!/usr/bin/env python3
"""Extract RoBERTa / EmoBERTa text features for MELD and IEMOCAP.

This script follows the text-only ERC preprocessing idea used by EmoBERTa:
for each current utterance, build a text input from optional past/future
utterances and optional speaker names, then export one utterance-level
embedding per sample.

No dataset files, model weights, or local absolute paths are included. Pass all
paths through CLI arguments.

Expected outputs:
    dict {
        "features": np.ndarray [N, hidden_dim],
        "labels":   np.ndarray [N],
        "ids":      np.ndarray [N]
    }

The resulting files are compatible with erc/datasets.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MELD_LABEL_MAP = {
    "neutral": 0,
    "joy": 1,
    "surprise": 2,
    "anger": 3,
    "sadness": 4,
    "disgust": 5,
    "fear": 6,
}

IEMOCAP_LABEL_MAP = {
    "neutral": 0,
    "neu": 0,
    "frustrated": 1,
    "fru": 1,
    "sadness": 2,
    "sad": 2,
    "anger": 3,
    "ang": 3,
    "excited": 4,
    "exc": 4,
    "happy": 5,
    "hap": 5,
}


@dataclass
class Utterance:
    uid: str
    dialogue_id: str
    utterance_index: int
    speaker: str
    text: str
    label: int


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_meld_split(raw_root: str, split: str) -> List[Utterance]:
    """Load one MELD split from MELD.Raw-style CSV files.

    The official raw split is named ``dev``; the exported feature split is often
    named ``val``. This function accepts either ``dev`` or ``val``.
    """
    raw_split = "dev" if split == "val" else split
    csv_path = Path(raw_root) / raw_split / f"{raw_split}_sent_emo.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    items: List[Utterance] = []
    for row in _read_csv(csv_path):
        emotion = row["Emotion"].strip().lower()
        if emotion not in MELD_LABEL_MAP:
            continue
        dia = int(row["Dialogue_ID"])
        utt = int(row["Utterance_ID"])
        items.append(
            Utterance(
                uid=f"dia{dia}_utt{utt}",
                dialogue_id=f"dia{dia}",
                utterance_index=utt,
                speaker=row.get("Speaker", "UNK").strip() or "UNK",
                text=row["Utterance"].strip(),
                label=MELD_LABEL_MAP[emotion],
            )
        )
    return sorted(items, key=lambda x: (x.dialogue_id, x.utterance_index))


def load_iemocap_split(csv_root: str, split: str) -> List[Utterance]:
    csv_path = Path(csv_root) / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IEMOCAP CSV not found: {csv_path}")

    items: List[Utterance] = []
    for order, row in enumerate(_read_csv(csv_path)):
        uid = row.get("id", "").strip()
        if not uid:
            continue
        emotion = row.get("emotion", row.get("label", "")).strip().lower()
        if emotion not in IEMOCAP_LABEL_MAP:
            continue

        dialogue_id = "_".join(uid.split("_")[:-1]) or uid
        if "speaker" in row and row["speaker"].strip():
            speaker = row["speaker"].strip()
        else:
            last = uid.rsplit("_", 1)[-1]
            speaker = last[0] if last else "UNK"

        if "start_time" in row and row["start_time"]:
            try:
                utterance_index = int(float(row["start_time"]) * 1000)
            except ValueError:
                utterance_index = order
        else:
            utterance_index = order

        items.append(
            Utterance(
                uid=uid,
                dialogue_id=dialogue_id,
                utterance_index=utterance_index,
                speaker=speaker,
                text=row.get("text", row.get("Utterance", "")).strip(),
                label=IEMOCAP_LABEL_MAP[emotion],
            )
        )
    return sorted(items, key=lambda x: (x.dialogue_id, x.utterance_index))


def group_by_dialogue(items: List[Utterance]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        groups.setdefault(item.dialogue_id, []).append(idx)
    return groups


def format_speaker(speaker: str, speaker_mode: str) -> str:
    if speaker_mode == "none":
        return ""
    if speaker_mode == "upper":
        return f"{speaker.upper()}: "
    if speaker_mode == "title":
        return f"{speaker.title()}: "
    if speaker_mode == "raw":
        return f"{speaker}: "
    raise ValueError(f"Unsupported speaker_mode: {speaker_mode}")


def build_text_input(
    items: List[Utterance],
    groups: Dict[str, List[int]],
    idx: int,
    num_past: int,
    num_future: int,
    speaker_mode: str,
    sep_token: str,
) -> str:
    """Build one RoBERTa input string from local conversational context."""
    item = items[idx]
    dlg_indices = groups[item.dialogue_id]
    pos = dlg_indices.index(idx)

    selected = []
    if num_past > 0:
        selected.extend(dlg_indices[max(0, pos - num_past):pos])
    selected.append(idx)
    if num_future > 0:
        selected.extend(dlg_indices[pos + 1:pos + 1 + num_future])

    segments: List[str] = []
    for j in selected:
        u = items[j]
        segments.append(f"{format_speaker(u.speaker, speaker_mode)}{u.text}")
    return f" {sep_token} ".join(segments)


def batch_iter(xs: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(xs), batch_size):
        yield xs[start:start + batch_size]


def encode_texts(
    texts: List[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    pooling: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    all_features: List[np.ndarray] = []
    for batch in tqdm(list(batch_iter(texts, batch_size)), desc="Encoding text"):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**encoded)
            hidden = out.last_hidden_state
            if pooling == "cls":
                feat = hidden[:, 0, :]
            elif pooling == "mean":
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                feat = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
        all_features.append(feat.detach().cpu().float().numpy())
    return np.concatenate(all_features, axis=0).astype(np.float32)


def output_split_name(dataset: str, split: str) -> str:
    if dataset == "meld" and split == "dev":
        return "val"
    return split


def process_split(args, split: str) -> None:
    if args.dataset == "meld":
        if not args.meld_root:
            raise ValueError("--meld_root is required for MELD")
        items = load_meld_split(args.meld_root, split)
    else:
        if not args.iemocap_csv_root:
            raise ValueError("--iemocap_csv_root is required for IEMOCAP")
        items = load_iemocap_split(args.iemocap_csv_root, split)

    groups = group_by_dialogue(items)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    sep_token = tokenizer.sep_token or "</s>"

    texts = [
        build_text_input(
            items=items,
            groups=groups,
            idx=i,
            num_past=args.num_past_utterances,
            num_future=args.num_future_utterances,
            speaker_mode=args.speaker_mode,
            sep_token=sep_token,
        )
        for i in range(len(items))
    ]

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    features = encode_texts(
        texts=texts,
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
    )

    ids = np.array([x.uid for x in items])
    labels = np.array([x.label for x in items], dtype=np.int64)

    os.makedirs(args.output_dir, exist_ok=True)
    out_split = output_split_name(args.dataset, split)
    out_path = Path(args.output_dir) / f"text_{out_split}.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"features": features, "labels": labels, "ids": ids}, f)

    print(f"Saved {len(items)} samples -> {out_path}")
    print(f"Feature shape: {features.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RoBERTa/EmoBERTa text embeddings for ERC.")
    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="HF model or local checkpoint, e.g. roberta-large or taewoonkim/emoberta-large")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="all",
                        help="train/dev/test/all for MELD, train/val/test/all for IEMOCAP")
    parser.add_argument("--meld_root", type=str, default="", help="Path to MELD.Raw root")
    parser.add_argument("--iemocap_csv_root", type=str, default="", help="Path containing IEMOCAP train/val/test CSV files")
    parser.add_argument("--speaker_mode", choices=["none", "upper", "title", "raw"], default="none",
                        help="Speaker-name formatting, following EmoBERTa-style speaker-aware inputs")
    parser.add_argument("--num_past_utterances", type=int, default=0)
    parser.add_argument("--num_future_utterances", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--gpu", type=int, default=0, help="Use -1 for CPU")
    args = parser.parse_args()

    if args.split == "all":
        splits = ["train", "dev", "test"] if args.dataset == "meld" else ["train", "val", "test"]
    else:
        splits = [args.split]

    for split in splits:
        process_split(args, split)


if __name__ == "__main__":
    main()
