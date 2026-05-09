import os
import random
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_HIDDEN_DIM = 512

LABELS: Dict[str, List[str]] = {
    "meld": ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"],
    "iemocap": ["neutral", "frustrated", "sadness", "anger", "excited", "happy"],
}


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def compute_metrics(preds, labels):
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
