# GitHub Release Checklist

## Removed or externalized

- [x] Local absolute paths
- [x] Usernames and machine-specific directory names
- [x] Raw dataset paths in source files
- [x] Checkpoints/model weights
- [x] Extracted `.pkl` features
- [x] Audio/video/image files
- [x] Training logs and predictions
- [x] Credentials

## Keep private

Do not commit:

- raw MELD/IEMOCAP media,
- speaker face libraries,
- extracted feature files,
- model checkpoints,
- local config files containing private paths,
- logs or prediction JSON files that contain sample-level outputs.

## Recommended GitHub workflow

```bash
git init
git add README.md RELEASE_CHECKLIST.md requirements.txt .gitignore erc feature_tools configs scripts
git status
git commit -m "Initial sanitized ERC release"
```

- [ ] Do not commit raw text CSV files, generated `text_*.pkl` features, or local HuggingFace checkpoints.
