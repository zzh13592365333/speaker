import csv
import json
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    dataset: str
    feature_root: str
    text_dir: str
    audio_dir: str
    video_dir: str
    vad_json: Optional[str] = None
    raw_csv_root: Optional[str] = None
    text_pattern: str = "text_{split}.pkl"
    audio_pattern: str = "audio_{split}.pkl"
    video_pattern: str = "video_{split}_aligned.pkl"
    split_alias: Optional[Dict[str, str]] = None

    @classmethod
    def from_dict(cls, cfg: Dict):
        paths = cfg.get("paths", cfg)
        return cls(
            dataset=cfg["dataset"].lower(),
            feature_root=paths["feature_root"],
            text_dir=paths.get("text_dir", "text"),
            audio_dir=paths.get("audio_dir", "audio"),
            video_dir=paths.get("video_dir", "video"),
            vad_json=paths.get("vad_json"),
            raw_csv_root=paths.get("raw_csv_root"),
            text_pattern=paths.get("text_pattern", "text_{split}.pkl"),
            audio_pattern=paths.get("audio_pattern", "audio_{split}.pkl"),
            video_pattern=paths.get("video_pattern", "video_{split}_aligned.pkl"),
            split_alias=paths.get("split_alias", {}),
        )


def _read_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class MultimodalDataset(Dataset):
    """Generic MELD/IEMOCAP dataset over pre-extracted text/audio/video features.

    Required pkl conventions:
      - text: dict with features, labels, ids
      - audio: either dict with features, labels, ids or list of {id_str, audio, label}
      - video: dict with features, labels, ids
    """

    def __init__(
        self,
        split: str,
        modes: List[str],
        cfg: DatasetConfig,
        window_size: int = 4,
        training: bool = False,
        context_mode: str = "past",
    ):
        if context_mode not in {"past", "full", "center"}:
            raise ValueError("context_mode must be one of: past, full, center")
        self.split = split
        self.raw_split = (cfg.split_alias or {}).get(split, split)
        self.modes = modes
        self.cfg = cfg
        self.window_size = window_size
        self.training = training
        self.context_mode = context_mode

        text_file = os.path.join(cfg.feature_root, cfg.text_dir, cfg.text_pattern.format(split=split))
        print(f"[{cfg.dataset.upper()} {split.upper()}] text: {text_file}")
        text_raw = _read_pickle(text_file)
        self.text_features = text_raw["features"]
        self.labels_arr = np.asarray(text_raw["labels"])
        self.ids_arr = np.asarray(text_raw["ids"]).astype(str)
        self.length = len(self.labels_arr)

        if len(self.text_features) != self.length or len(self.ids_arr) != self.length:
            raise ValueError(f"{split}: text/features, labels, ids length mismatch")

        self.audio_map = {}
        if "audio" in modes:
            audio_file = os.path.join(cfg.feature_root, cfg.audio_dir, cfg.audio_pattern.format(split=split))
            print(f"[{cfg.dataset.upper()} {split.upper()}] audio: {audio_file}")
            self.audio_map = self._load_audio_map(audio_file)

        self.video_map = {}
        if "video" in modes:
            video_file = os.path.join(cfg.feature_root, cfg.video_dir, cfg.video_pattern.format(split=split))
            print(f"[{cfg.dataset.upper()} {split.upper()}] video: {video_file}")
            video_raw = _read_pickle(video_file)
            self.video_map = {str(i): f for i, f in zip(video_raw["ids"], video_raw["features"])}

        self._check_modal_coverage()
        self.dialogue_ids = [self._parse_dialogue_id(str(i)) for i in self.ids_arr]
        self.speakers = self._load_speakers()
        unique_dlg = sorted(set(self.dialogue_ids))
        self.dlg2idx = {d: i for i, d in enumerate(unique_dlg)}
        self.vad_data = self._load_vad_json(cfg.vad_json)
        self.vad_arr = self._build_vad_array()
        print(f"[{cfg.dataset.upper()} {split.upper()}] samples={self.length}, context_mode={context_mode}")

    def _load_audio_map(self, path: str) -> Dict[str, object]:
        raw = _read_pickle(path)
        if isinstance(raw, dict) and all(k in raw for k in ("features", "ids")):
            return {str(i): f for i, f in zip(raw["ids"], raw["features"])}
        if isinstance(raw, list):
            return {str(item.get("id_str", item.get("id"))): item.get("audio", item.get("feature")) for item in raw}
        raise ValueError(f"Unsupported audio pkl format: {path}")

    def _check_modal_coverage(self) -> None:
        text_ids = list(map(str, self.ids_arr))
        text_set = set(text_ids)
        for name, mp in (("audio", self.audio_map), ("video", self.video_map)):
            if name not in self.modes:
                continue
            modal_set = set(map(str, mp.keys()))
            missing = [i for i in text_ids if i not in modal_set]
            extra = [i for i in modal_set if i not in text_set]
            print(f"[{self.cfg.dataset.upper()} {self.split.upper()}] {name}: text={len(text_ids)}, {name}={len(modal_set)}, missing={len(missing)}, extra={len(extra)}")
            if missing:
                raise ValueError(f"{self.split}: missing {name} features for {len(missing)} text ids, e.g. {missing[:5]}")

    def _parse_dialogue_id(self, id_str: str) -> str:
        if self.cfg.dataset == "meld":
            return id_str.rsplit("_utt", 1)[0]
        return id_str.rsplit("_", 1)[0]

    def _parse_speaker_from_id(self, id_str: str) -> str:
        if self.cfg.dataset == "iemocap":
            last = id_str.rsplit("_", 1)[-1]
            return last[0] if last else "UNK"
        return "UNK"

    def _load_speakers(self) -> List[str]:
        if self.cfg.dataset != "meld" or not self.cfg.raw_csv_root:
            return [self._parse_speaker_from_id(str(i)) for i in self.ids_arr]
        csv_name = f"{self.raw_split}_sent_emo.csv"
        csv_path = os.path.join(self.cfg.raw_csv_root, self.raw_split, csv_name)
        speaker_map = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        key = f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
                        speaker_map[key] = row["Speaker"].strip()
                    except Exception:
                        continue
        return [speaker_map.get(str(i), "UNK") for i in self.ids_arr]

    @staticmethod
    def _load_vad_json(path: Optional[str]) -> Dict:
        if not path:
            return {}
        if not os.path.exists(path):
            print(f"[WARN] VAD json not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_vad_array(self) -> np.ndarray:
        arr = np.zeros((self.length, 3), dtype=np.float32)
        for i, uid in enumerate(self.ids_arr):
            if self.cfg.dataset == "meld":
                key = f"meld_{self.raw_split}_{uid}"
            else:
                key = f"iemocap_{uid}"
            entry = self.vad_data.get(key)
            if entry is not None:
                arr[i] = entry.get("vad_vector", [0.0, 0.0, 0.0])
        return arr

    def _augment_feature(self, feat, noise_scale=0.05, mask_prob=0.1):
        if not self.training:
            return feat
        feat = feat.float() + torch.randn_like(feat.float()) * noise_scale
        if feat.dim() == 2 and random.random() < mask_prob:
            seq_len = feat.shape[0]
            mask_len = max(1, int(seq_len * 0.15))
            start = random.randint(0, max(0, seq_len - mask_len))
            feat[start:start + mask_len] = 0
        return feat

    def _get_text_tensor(self, idx: int) -> torch.Tensor:
        t = torch.as_tensor(_as_numpy(self.text_features[idx]), dtype=torch.float32)
        while t.dim() > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.dim() == 2:
            t = t[0]
        if t.dim() != 1 or t.shape[-1] != 1024:
            raise ValueError(f"Invalid text feature shape at idx={idx}: {tuple(t.shape)}")
        return t

    def _get_raw_features(self, idx: int) -> Dict[str, torch.Tensor]:
        uid = str(self.ids_arr[idx])
        feat = {"text_feat": self._get_text_tensor(idx), "vad": torch.tensor(self.vad_arr[idx], dtype=torch.float32)}

        if "audio" in self.modes:
            a = torch.as_tensor(_as_numpy(self.audio_map[uid]), dtype=torch.float32)
            if a.dim() != 2 or a.shape[-1] != 1024:
                raise ValueError(f"Invalid audio feature shape for {uid}: {tuple(a.shape)}")
            feat["audio"] = self._augment_feature(a, noise_scale=0.05)
        else:
            feat["audio"] = torch.zeros(300, 1024, dtype=torch.float32)

        if "video" in self.modes:
            v = torch.as_tensor(_as_numpy(self.video_map[uid]), dtype=torch.float32)
            if v.dim() == 1:
                v = v.unsqueeze(0)
            elif v.dim() == 2 and v.shape[0] != 1:
                v = v[-1:, :]
            if v.dim() != 2 or v.shape[-1] != 4096:
                raise ValueError(f"Invalid video feature shape for {uid}: {tuple(v.shape)}")
            feat["video"] = self._augment_feature(v, noise_scale=0.03)
        else:
            feat["video"] = torch.zeros(1, 4096, dtype=torch.float32)
        return feat

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        curr = self._get_raw_features(idx)
        ret = {
            "text_feat": curr["text_feat"],
            "audio": curr["audio"],
            "video": curr["video"],
            "vad": curr["vad"],
            "label": torch.tensor(int(self.labels_arr[idx]), dtype=torch.long),
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "video_id_idx": torch.tensor(self.dlg2idx.get(self.dialogue_ids[idx], 0), dtype=torch.long),
        }
        curr_spk = self.speakers[idx]

        def neighbor(target_idx):
            if target_idx < 0 or target_idx >= self.length:
                return None, 2
            if self.dialogue_ids[idx] != self.dialogue_ids[target_idx]:
                return None, 2
            f = self._get_raw_features(target_idx)
            data = {
                "text": f["text_feat"],
                "audio": f["audio"].mean(dim=0),
                "video": f["video"].squeeze(0),
                "vad": f["vad"],
            }
            rel = 0 if self.speakers[target_idx] == curr_spk else 1
            return data, rel

        def collect(start, end, step):
            texts, audios, videos, spks, masks, vads = [], [], [], [], [], []
            for offset in range(start, end, step):
                data, rel = neighbor(idx + offset)
                if data is None:
                    texts.append(torch.zeros_like(ret["text_feat"]))
                    audios.append(torch.zeros(1024))
                    videos.append(torch.zeros(4096))
                    spks.append(torch.tensor(2, dtype=torch.long))
                    masks.append(torch.tensor(0.0))
                    vads.append(torch.zeros(3))
                else:
                    texts.append(data["text"])
                    audios.append(data["audio"])
                    videos.append(data["video"])
                    spks.append(torch.tensor(rel, dtype=torch.long))
                    masks.append(torch.tensor(1.0))
                    vads.append(data["vad"])
            return torch.stack(texts), torch.stack(audios), torch.stack(videos), torch.stack(spks), torch.stack(masks), torch.stack(vads)

        p_t, p_a, p_v, p_s, p_m, p_vad = collect(-self.window_size, 0, 1)
        ret.update({"prev_text": p_t, "prev_audio": p_a, "prev_video": p_v, "prev_spk": p_s, "prev_mask": p_m, "prev_vad": p_vad})

        if self.context_mode in {"full", "center"}:
            n_t, n_a, n_v, n_s, n_m, n_vad = collect(1, self.window_size + 1, 1)
        else:
            n_t = torch.zeros(self.window_size, 1024)
            n_a = torch.zeros(self.window_size, 1024)
            n_v = torch.zeros(self.window_size, 4096)
            n_s = torch.full((self.window_size,), 2, dtype=torch.long)
            n_m = torch.zeros(self.window_size)
            n_vad = torch.zeros(self.window_size, 3)
        ret.update({"next_text": n_t, "next_audio": n_a, "next_video": n_v, "next_spk": n_s, "next_mask": n_m, "next_vad": n_vad})
        return ret


def mm_collate_fn(batch):
    out = {
        "labels": torch.stack([x["label"] for x in batch]),
        "vad": torch.stack([x["vad"] for x in batch]),
        "sample_idx": torch.stack([x["sample_idx"] for x in batch]),
        "text": torch.stack([x["text_feat"] for x in batch]).unsqueeze(1),
        "prev_text": torch.stack([x["prev_text"] for x in batch]),
        "next_text": torch.stack([x["next_text"] for x in batch]),
        "prev_audio": torch.stack([x["prev_audio"] for x in batch]),
        "next_audio": torch.stack([x["next_audio"] for x in batch]),
        "prev_video": torch.stack([x["prev_video"] for x in batch]),
        "next_video": torch.stack([x["next_video"] for x in batch]),
        "prev_spk": torch.stack([x["prev_spk"] for x in batch]),
        "next_spk": torch.stack([x["next_spk"] for x in batch]),
        "prev_mask": torch.stack([x["prev_mask"] for x in batch]),
        "next_mask": torch.stack([x["next_mask"] for x in batch]),
        "prev_vad": torch.stack([x["prev_vad"] for x in batch]),
        "next_vad": torch.stack([x["next_vad"] for x in batch]),
        "video_id_idx": torch.stack([x["video_id_idx"] for x in batch]),
    }
    out["audio"] = pad_sequence([x["audio"] for x in batch], batch_first=True, padding_value=0.0)
    videos = pad_sequence([x["video"] for x in batch], batch_first=True, padding_value=0.0)
    out["video"] = videos
    out["video_mask"] = (torch.sum(torch.abs(videos), dim=-1) == 0)
    return out
