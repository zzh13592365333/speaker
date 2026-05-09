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
# Keep this order consistent with configs/iemocap.example.json and erc/utils.py.
IEMOCAP_EMOTIONS = ["neutral", "frustrated", "sadness", "anger", "excited", "happy"]
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


def sample_first_ratio_frames(frame_paths, max_frames=5, first_ratio=0.70):
    """Uniformly sample frames from the first portion of an utterance clip.

    The paper uses frames sampled from the first 70% of the current utterance to
    reduce boundary contamination from the next utterance.
    """
    frame_paths = list(sorted(frame_paths))
    if not frame_paths:
        return []
    keep_n = max(1, int(np.ceil(len(frame_paths) * first_ratio)))
    frame_paths = frame_paths[:keep_n]
    if len(frame_paths) <= max_frames:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, max_frames).round().astype(int)
    return [frame_paths[i] for i in idx]


def reference_image(face_library, speaker):
    if not face_library or not speaker:
        return None
    d = Path(face_library) / speaker
    if not d.exists():
        return None
    return next((str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)


def iemocap_dialogue_id(sample_id):
    # Common IEMOCAP ids look like Ses01F_impro01_F000. The final segment is the utterance index.
    return str(sample_id).rsplit("_", 1)[0]


def build_dialogue_context(df, row_pos, dataset, context_window=4):
    """Build a past-only dialogue context ending at the current utterance."""
    row = df.iloc[row_pos]
    if dataset == "meld":
        dlg_id = row["dialogue_id"]
        same_dialogue = df[df["dialogue_id"] == dlg_id]
        same_dialogue = same_dialogue[same_dialogue.index <= row.name].tail(context_window + 1)
    else:
        dlg_id = iemocap_dialogue_id(row["id"])
        same_dialogue = df[df["id"].astype(str).map(iemocap_dialogue_id) == dlg_id]
        same_dialogue = same_dialogue[same_dialogue.index <= row.name].tail(context_window + 1)

    lines = []
    for _, r in same_dialogue.iterrows():
        spk = str(r.get("speaker", "Speaker"))
        utt = str(r.get("text", ""))
        lines.append(f"{spk}: {utt}")
    return "\n".join(lines)


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


def build_prompt(args, row, speaker, has_ref, has_frames, dialogue_context, audio_text, vad_prompt):
    emotion_names = MELD_EMOTIONS if args.dataset == "meld" else IEMOCAP_EMOTIONS
    label_space = ", ".join(emotion_names)

    prompt = []

    # PGVI: visual inputs + task formulation.
    if not args.drop_speaker:
        prompt.append(
            f"The first image is a reference photo of the current speaker [{speaker}]. "
            f"Please use it to locate [{speaker}] in the subsequent video frames, and focus on "
            f"[{speaker}]'s facial expressions, body language, and gestures."
        )
        prompt.append(
            "The reference photo may not perfectly match the in-show appearance due to lighting "
            "or styling differences. Use it as a general identity guide rather than an exact visual match."
        )

    prompt.append("You are an expert in emotion recognition from conversations.")
    if args.dataset == "meld":
        prompt.append("This clip is from a sitcom with a studio audience.")
    if not args.drop_speaker:
        prompt.append(
            f"The target speaker for this sample is [{speaker}]. Always analyze the emotion of "
            f"[{speaker}] only. Never switch your analysis to another visible person in the scene."
        )

    if has_frames:
        prompt.append(
            f"The video frames are arranged in chronological order and are sampled from the first "
            f"{int(args.first_frame_ratio * 100)}% of the utterance clip to reduce boundary contamination "
            "from the next utterance. If there is a sudden scene cut where the person or background "
            "changes drastically between frames, give less importance to the frames after the cut because "
            "they may belong to the next utterance."
        )
        if args.dataset == "meld":
            prompt.append(
                "Ignore canned laughter and background audience reactions, and focus on the target "
                "speaker's expression, intent, and visual behavior."
            )

    prompt.append(
        f"The following candidate emotion names define the task label space only: {label_space}. "
        "Do not assume that any candidate emotion is the correct label."
    )

    # ASGI: affective semantic guidance.
    if not args.drop_context and dialogue_context:
        prompt.append(f"Dialogue context:\n{dialogue_context}")
        prompt.append(f"Target speaker [{speaker}] says: \"{row.get('text', '')}\"")

    if audio_text:
        prompt.append(f"Acoustic characteristics of [{speaker}]'s voice: {audio_text}")

    if vad_prompt:
        prompt.append(vad_prompt)

    prompt.append(
        f"Based on {evidence_phrase(has_frames, not args.drop_context and bool(dialogue_context), bool(vad_prompt), bool(audio_text), None if args.drop_speaker else speaker)}, "
        "determine the speaker's emotion."
    )
    return "\n".join(prompt)


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
    ap.add_argument("--split", choices=["train", "val", "dev", "test", "all"], default="val",
                    help="Use train/val/test for IEMOCAP. MELD raw validation split is dev; passing val maps to dev for MELD.")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_script_dir", default="", help="Directory containing qwen3_vl_embedding.py if needed")
    ap.add_argument("--csv_root", required=True)
    ap.add_argument("--frame_root", required=True)
    ap.add_argument("--face_library", default="")
    ap.add_argument("--audio_desc_json", default="")
    ap.add_argument("--vad_json", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=5)
    ap.add_argument("--first_frame_ratio", type=float, default=0.70)
    ap.add_argument("--context_window", type=int, default=4)
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

    splits = (["train", "dev", "test"] if args.dataset == "meld" else ["train", "val", "test"]) if args.split == "all" else [args.split]
    for split in splits:
        if args.dataset == "meld":
            raw_split = "dev" if split == "val" else split
            csv_path = Path(args.csv_root) / raw_split / f"{raw_split}_sent_emo.csv"
            df = pd.read_csv(csv_path).rename(columns={"Utterance": "text", "Speaker": "speaker", "Emotion": "emotion", "Dialogue_ID": "dialogue_id", "Utterance_ID": "utterance_id"})
        else:
            if split == "dev":
                raise ValueError("IEMOCAP uses the split name 'val' instead of 'dev'. Please pass --split val.")
            raw_split = split
            csv_path = Path(args.csv_root) / f"{raw_split}.csv"
            df = pd.read_csv(csv_path)

        feats, ids, labels = [], [], []
        for row_pos, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"{args.dataset}-{split}")):
            if args.dataset == "meld":
                did, uid = int(row["dialogue_id"]), int(row["utterance_id"])
                sample_id = f"dia{did}_utt{uid}"
                lookup_id = f"meld_{raw_split}_{sample_id}"
                speaker = str(row["speaker"])
                frame_dir = Path(args.frame_root) / raw_split / f"{did:04d}" / f"{uid:02d}"
                emotion = str(row["emotion"]).lower()
            else:
                sample_id = str(row["id"])
                lookup_id = f"iemocap_{sample_id}"
                speaker = str(row.get("speaker", "speaker"))
                frame_dir = Path(args.frame_root) / raw_split / sample_id
                emotion = str(row["emotion"]).lower()
            if emotion not in LABEL_MAP[args.dataset]:
                continue

            all_frame_paths = valid_images(sorted(frame_dir.glob("*.jpg")) if frame_dir.exists() else [])
            frame_paths = sample_first_ratio_frames(all_frame_paths, max_frames=args.max_frames, first_ratio=args.first_frame_ratio)
            ref = None if args.drop_speaker else reference_image(args.face_library, speaker)
            image_paths = ([ref] if ref else []) + frame_paths

            dialogue_context = "" if args.drop_context else build_dialogue_context(df, row_pos, args.dataset, context_window=args.context_window)

            use_audio = (not args.drop_audio_desc) and lookup_id in audio_desc
            audio_text = str(audio_desc[lookup_id]).strip() if use_audio else ""
            if audio_text.upper() == "ERROR":
                audio_text = ""

            use_vad = (not args.drop_vad) and lookup_id in vad_desc and vad_desc[lookup_id].get("prompt")
            vad_prompt = vad_desc[lookup_id]["prompt"] if use_vad else ""

            prompt_text = build_prompt(args, row, speaker, bool(ref), bool(frame_paths), dialogue_context, audio_text, vad_prompt)
            messages = [{"role": "user", "content": [*[{"type": "image", "image": p} for p in image_paths], {"type": "text", "text": prompt_text}]}]
            try:
                feats.append(extract_feature(messages, image_paths, processor, model, device))
                ids.append(sample_id)
                labels.append(LABEL_MAP[args.dataset][emotion])
            except Exception as e:
                print(f"[WARN] failed {sample_id}: {e}")
        os.makedirs(args.output_dir, exist_ok=True)
        out_split = "val" if split == "dev" else split
        out_path = Path(args.output_dir) / f"video_{out_split}_aligned.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"features": np.asarray(feats), "ids": np.asarray(ids), "labels": np.asarray(labels)}, f)
        print(f"Saved {len(ids)} samples -> {out_path}")


if __name__ == "__main__":
    main()
