"""Shared helpers for the training and evaluation scripts."""

import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(name: str):
    """Return (texts, label_ids) for 'train' | 'val' | 'test'."""
    rows = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
    return [r["text"] for r in rows], [r["label_id"] for r in rows]


def load_labels() -> list[str]:
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))


def save_predictions(model_key: str, y_true, y_pred, y_conf) -> None:
    """Persist test-set predictions so evaluate.py can build every plot from
    one place rather than re-running training."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"preds_{model_key}.json").write_text(
        json.dumps(
            {
                "y_true": [int(v) for v in y_true],
                "y_pred": [int(v) for v in y_pred],
                "y_conf": [float(v) for v in y_conf],
            }
        ),
        encoding="utf-8",
    )


def update_metrics(model_key: str, payload: dict) -> None:
    """Merge one model's numbers into results/metrics.json."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "metrics.json"
    all_metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_metrics[model_key] = payload
    path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
