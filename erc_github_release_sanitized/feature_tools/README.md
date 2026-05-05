# Feature Tools

All scripts use command-line paths instead of private machine-specific constants.

Typical order:

0. Extract text features with RoBERTa / EmoBERTa-style inputs:
```bash
python feature_tools/extract_text_emoberta_features.py \
  --dataset meld \
  --meld_root /path/to/MELD.Raw \
  --model_name_or_path roberta-large \
  --output_dir /path/to/meld/features/wenben_sorted \
  --split all \
  --speaker_mode none \
  --num_past_utterances 0 \
  --num_future_utterances 0
```

The `speaker_mode`, `num_past_utterances`, and `num_future_utterances` options mirror the text-context settings used in the public EmoBERTa ERC codebase.

1. Build lexical VAD prompts:
```bash
python feature_tools/preprocess_vad.py \
  --vad_lexicon /path/to/NRC-VAD-Lexicon.txt \
  --meld_root /path/to/MELD.Raw \
  --iemocap_csv_root /path/to/iemocap/csv \
  --output_json /path/to/all_vad.json
```

2. Generate acoustic descriptions with Qwen2-Audio:
```bash
python feature_tools/extract_audio_descriptions.py \
  --model_path /path/to/Qwen2-Audio-7B-Instruct \
  --meld_root /path/to/MELD.Raw \
  --iemocap_root /path/to/IEMOCAP_full_release \
  --output_dir outputs/audio_desc \
  --rank 0 --world_size 1
```

3. Extract MELD emotion2vec audio features:
```bash
python feature_tools/extract_emotion2vec_meld.py \
  --emotion2vec_repo /path/to/emotion2vec-main \
  --checkpoint_path /path/to/emotion2vec_plus_large/model.pt \
  --meld_root /path/to/MELD.Raw \
  --output_dir /path/to/meld/features/yinpin_emotion2vec \
  --split all --gpu 0
```

For VLM visual features, keep the same policy: pass `--model_path`, `--frame_root`, `--csv_root`, `--face_library`, `--vad_json`, and `--audio_desc_json` through CLI/config instead of committing local absolute paths.
