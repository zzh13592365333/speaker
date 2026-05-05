"""Build lexical VAD prompts for MELD and/or IEMOCAP without hard-coded local paths."""
import argparse
import csv
import json
import os
import re
from pathlib import Path

STOPWORDS = {"i","me","my","we","our","you","your","he","him","his","she","her","it","they","them","the","a","an","and","but","or","of","to","in","on","is","are","was","were","be","do","does","did","have","has","had","not","no","yeah","okay","oh","um","uh"}
INTERJECTION_VAD = {"wow": (0.75,0.80,0.60), "yay": (0.85,0.75,0.65), "ugh": (0.20,0.70,0.40), "oh": (0.55,0.40,0.50), "whoa": (0.60,0.85,0.50), "yikes": (0.20,0.80,0.35), "damn": (0.20,0.75,0.50)}
NEGATION_WORDS = {"not","no","never","neither","nor","nobody","nothing","nowhere","hardly","barely","scarcely","without","don't","doesn't","didn't","won't","wouldn't","can't","couldn't","shouldn't","isn't","aren't","wasn't","weren't"}


def load_vad_lexicon(path):
    vad = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                word, v, a, d = parts
                vad[word.lower()] = (float(v), float(a), float(d))
    return vad


def tokenize(text):
    return re.findall(r"\b[a-z][a-z']*[a-z]\b|\b[a-z]\b", text.lower())


def weight(v, a, d):
    return max(abs(v - 0.5) + abs(a - 0.5) + abs(d - 0.5), 0.05)


def compute_vad(text, lexicon):
    tokens = tokenize(text)
    negated = [False] * len(tokens)
    for i, tok in enumerate(tokens):
        if tok in NEGATION_WORDS:
            for j in range(i + 1, min(i + 4, len(tokens))):
                negated[j] = True
    hits = []
    for pass_id in [0, 1]:
        for i, tok in enumerate(tokens):
            if pass_id == 0 and tok in STOPWORDS:
                continue
            if tok in INTERJECTION_VAD:
                v, a, d = INTERJECTION_VAD[tok]
            elif tok in lexicon:
                v, a, d = lexicon[tok]
            else:
                continue
            if negated[i]:
                v = 1.0 - v
            hits.append((v, a, d))
        if hits:
            break
    if not hits:
        return {"valence":0.5,"arousal":0.5,"dominance":0.5,"matched_words":0,"used_fallback":False,"prompt":"","vad_vector":[0.5,0.5,0.5]}
    ws = [weight(*h) for h in hits]
    total = sum(ws)
    vals = [round(sum(h[k] * w for h, w in zip(hits, ws)) / total, 4) for k in range(3)]
    prompt = f"Lexical emotion cues from the transcript: valence={vals[0]:.2f}, arousal={vals[1]:.2f}, dominance={vals[2]:.2f}"
    return {"valence":vals[0],"arousal":vals[1],"dominance":vals[2],"matched_words":len(hits),"used_fallback":False,"prompt":prompt,"vad_vector":vals}


def process_meld(meld_root, lexicon, out):
    valid = {"neutral","joy","surprise","anger","sadness","disgust","fear"}
    for split in ["train", "dev", "test"]:
        csv_path = Path(meld_root) / split / f"{split}_sent_emo.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                emo = row["Emotion"].strip().lower()
                if emo not in valid:
                    continue
                uid = f"meld_{split}_dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"
                item = compute_vad(row["Utterance"].strip(), lexicon)
                item.update({"dataset":"meld", "text": row["Utterance"].strip()})
                out[uid] = item


def process_iemocap(iemocap_csv_root, lexicon, out):
    for split in ["train", "dev", "test"]:
        csv_path = Path(iemocap_csv_root) / f"{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                uid = f"iemocap_{row['id'].strip()}"
                item = compute_vad(row["text"].strip(), lexicon)
                item.update({"dataset":"iemocap", "text": row["text"].strip()})
                out[uid] = item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad_lexicon", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--meld_root", default="")
    ap.add_argument("--iemocap_csv_root", default="")
    args = ap.parse_args()
    lexicon = load_vad_lexicon(args.vad_lexicon)
    results = {}
    if args.meld_root:
        process_meld(args.meld_root, lexicon, results)
    if args.iemocap_csv_root:
        process_iemocap(args.iemocap_csv_root, lexicon, results)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} VAD entries to {args.output_json}")


if __name__ == "__main__":
    main()
