"""Generate acoustic descriptions with Qwen2-Audio. All paths are CLI arguments."""
import argparse
import glob
import json
import math
import os
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

PROMPT = """Listen carefully to this audio clip. Focus only on the main speaker's voice and ignore background noise, audience laughter, music, or crowd sounds.
In exactly 2 complete English sentences, describe what is emotionally distinctive about the main speaker's voice.
Focus on pitch breaks, breathiness, vocal fry, clipped words, pauses, strained tension, rushed rhythm, or flat affect. Respond in English only."""


def collect_meld(meld_root):
    data = []
    for split in ["train", "dev", "test"]:
        base = Path(meld_root) / split / f"{split}_splits"
        for wav in base.glob("*.wav") if base.exists() else []:
            data.append((f"meld_{split}_{wav.stem}", str(wav)))
    return data


def collect_iemocap(iemocap_root):
    data = []
    for wav in Path(iemocap_root).glob("Session*/sentences/wav/*/*.wav"):
        # Keep the key format consistent with the VLM extractor and VAD files:
        #   iemocap_<utterance_id>
        # where <utterance_id> is the wav stem, e.g. Ses01F_impro01_F000.
        data.append((f"iemocap_{wav.stem}", str(wav)))
    return data


def infer(processor, model, wav_path):
    speech, sr = librosa.load(wav_path, sr=processor.feature_extractor.sampling_rate)
    msg = [{"role":"user", "content":[{"type":"audio", "audio_url":wav_path}, {"type":"text", "text":PROMPT}]}]
    text = processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=speech, return_tensors="pt", sampling_rate=sr).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, temperature=0.7, top_p=0.9, do_sample=True)
    out = out[:, inputs.input_ids.size(1):]
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def merge(output_dir, output_json):
    merged = {}
    for fp in sorted(glob.glob(os.path.join(output_dir, "audio_desc_rank*.json"))):
        with open(fp, encoding="utf-8") as f:
            merged.update(json.load(f))
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(merged)} entries -> {output_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--meld_root", default="")
    ap.add_argument("--iemocap_root", default="")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.merge:
        merge(args.output_dir, os.path.join(args.output_dir, "audio_desc_merged.json")); return
    device_id = args.rank % max(1, torch.cuda.device_count())
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map={"":device_id}).eval()
    all_data = []
    if args.meld_root: all_data += collect_meld(args.meld_root)
    if args.iemocap_root: all_data += collect_iemocap(args.iemocap_root)
    chunk = math.ceil(len(all_data) / args.world_size)
    local = all_data[args.rank * chunk:min((args.rank + 1) * chunk, len(all_data))]
    out_path = os.path.join(args.output_dir, f"audio_desc_rank{args.rank}.json")
    results = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    for uid, path in tqdm(local, desc=f"rank {args.rank}"):
        if uid not in results:
            try:
                results[uid] = infer(processor, model, path)
            except Exception as e:
                results[uid] = "ERROR"
        if len(results) % 50 == 0:
            json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(results, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
