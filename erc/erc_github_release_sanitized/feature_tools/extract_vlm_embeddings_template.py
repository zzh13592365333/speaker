"""Template for extracting Qwen3-VL-Embedding features.

This file intentionally contains no local absolute paths. Fill paths by CLI.
It mirrors the paper setting: frames + optional speaker reference + dialogue/audio/VAD prompts.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
IEMOCAP_EMOTIONS = ["anger", "frustrated", "excited", "happy", "sadness", "neutral"]
LABEL_MAP = {"meld": {e: i for i, e in enumerate(MELD_EMOTIONS)}, "iemocap": {e: i for i, e in enumerate(IEMOCAP_EMOTIONS)}}


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def valid_images(paths):
    out = []
    for p in paths:
        try:
            with Image.open(str(p)) as img:
                img.verify()
            out.append(str(p))
        except Exception:
            pass
    return out


def reference_image(face_library, speaker):
    if not face_library or not speaker:
        return None
    d = Path(face_library) / speaker
    if not d.exists():
        return None
    return next((str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)


def evidence_phrase(has_image, use_context, use_vad, use_audio, speaker=None):
    parts = []
    if has_image:
        parts.append(f"the visual cues of [{speaker}]" if speaker else "the visual cues")
    if use_context:
        parts.append("the dialogue context")
    if use_vad:
        parts.append("lexical emotion cues")
    if use_audio:
        parts.append(f"the acoustic characteristics of [{speaker}]'s voice" if speaker else "the acoustic characteristics of the speaker's voice")
    return ", ".join(parts) if parts else "the available information"


def extract_feature(messages, image_paths, processor, model, device):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = []
    for p in image_paths:
        with Image.open(p) as img:
            images.append(img.convert("RGB").copy())
    inputs = processor(text=[text], images=images if images else None, padding=False, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            attn = inputs["attention_mask"]
            pos = (attn.shape[1] - attn.flip(dims=[1]).argmax(dim=1) - 1).clamp(0, attn.shape[1] - 1)
            feat = hidden[torch.arange(attn.shape[0], device=hidden.device), pos, :]
        else:
            feat = hidden[:, -1, :]
    return feat[0].cpu().float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="dev")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_script_dir", default="", help="Directory containing qwen3_vl_embedding.py if needed")
    ap.add_argument("--csv_root", required=True)
    ap.add_argument("--frame_root", required=True)
    ap.add_argument("--face_library", default="")
    ap.add_argument("--audio_desc_json", default="")
    ap.add_argument("--vad_json", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--drop_vad", action="store_true")
    ap.add_argument("--drop_context", action="store_true")
    ap.add_argument("--drop_audio_desc", action="store_true")
    ap.add_argument("--drop_speaker", action="store_true", help="Remove speaker name, target-speaker constraint, and reference image")
    args = ap.parse_args()

    if args.model_script_dir:
        sys.path.insert(0, args.model_script_dir)
    from qwen3_vl_embedding import Qwen3VLForEmbedding

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=256 * 32 * 32, max_pixels=512 * 32 * 32)
    model = Qwen3VLForEmbedding.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device).eval()
    audio_desc = load_json(args.audio_desc_json)
    vad_desc = load_json(args.vad_json)

    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    for split in splits:
        if args.dataset == "meld":
            csv_path = Path(args.csv_root) / split / f"{split}_sent_emo.csv"
            df = pd.read_csv(csv_path).rename(columns={"Utterance":"text", "Speaker":"speaker", "Emotion":"emotion", "Dialogue_ID":"dialogue_id", "Utterance_ID":"utterance_id"})
        else:
            csv_path = Path(args.csv_root) / f"{split}.csv"
            df = pd.read_csv(csv_path)

        feats, ids, labels = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.dataset}-{split}"):
            if args.dataset == "meld":
                did, uid = int(row["dialogue_id"]), int(row["utterance_id"])
                sample_id = f"dia{did}_utt{uid}"
                lookup_id = f"meld_{split}_{sample_id}"
                speaker = str(row["speaker"])
                frame_dir = Path(args.frame_root) / split / f"{did:04d}" / f"{uid:02d}"
                emotion = str(row["emotion"]).lower()
            else:
                sample_id = str(row["id"])
                lookup_id = f"iemocap_{sample_id}"
                speaker = str(row.get("speaker", "speaker"))
                frame_dir = Path(args.frame_root) / split / sample_id
                emotion = str(row["emotion"]).lower()
            if emotion not in LABEL_MAP[args.dataset]:
                continue
            frame_paths = valid_images(sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else [])
            ref = None if args.drop_speaker else reference_image(args.face_library, speaker)
            image_paths = ([ref] if ref else []) + frame_paths

            prompt = ["You are an expert in emotion recognition from conversations."]
            if not args.drop_speaker:
                prompt.append(f"The target speaker is [{speaker}]. Always analyze [{speaker}] only.")
            if ref:
                prompt.append(f"The first image is a reference photo of [{speaker}]. Use it to locate the target speaker in subsequent frames.")
            if frame_paths:
                prompt.append("Frames are chronological and sampled from the utterance clip. Focus on facial expressions, body language, and gestures.")
            if not args.drop_context and "text" in row:
                prompt.append(f"Current utterance: \"{row['text']}\"")
            use_audio = (not args.drop_audio_desc) and lookup_id in audio_desc
            if use_audio:
                prompt.append(f"Acoustic characteristics: {audio_desc[lookup_id]}")
            use_vad = (not args.drop_vad) and lookup_id in vad_desc and vad_desc[lookup_id].get("prompt")
            if use_vad:
                prompt.append(vad_desc[lookup_id]["prompt"])
            prompt.append(f"Based on {evidence_phrase(bool(frame_paths), not args.drop_context, use_vad, use_audio, None if args.drop_speaker else speaker)}, determine the speaker's emotion.")
            messages = [{"role":"user", "content":[*[{"type":"image", "image":p} for p in image_paths], {"type":"text", "text":" ".join(prompt)}]}]
            try:
                feats.append(extract_feature(messages, image_paths, processor, model, device))
                ids.append(sample_id); labels.append(LABEL_MAP[args.dataset][emotion])
            except Exception as e:
                print(f"[WARN] failed {sample_id}: {e}")
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = Path(args.output_dir) / f"video_{'val' if split == 'dev' else split}_aligned.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"features": np.asarray(feats), "ids": np.asarray(ids), "labels": np.asarray(labels)}, f)
        print(f"Saved {len(ids)} samples -> {out_path}")


if __name__ == "__main__":
    main()
