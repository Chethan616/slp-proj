"""Dump predictions for one split using the saved fine-tuned model.

Used to produce validation-set predictions, so that the out-of-scope confidence
threshold can be selected on validation and only *reported* on test. Selecting it
on the test split would leak test information into a deployed hyperparameter.

    python train/predict_split.py val
"""

import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, load_split, save_predictions

MODEL_DIR = ROOT / "app" / "model"
MAX_LEN = 32
BATCH = 64


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    texts, y_true = load_split(split)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(
                texts[i : i + BATCH],
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            probs = torch.softmax(model(**enc).logits, dim=-1)
            c, p = probs.max(dim=-1)
            preds.extend(p.tolist())
            confs.extend(c.tolist())

    acc = sum(int(a == b) for a, b in zip(preds, y_true)) / len(y_true)
    save_predictions(f"distilbert_{split}", y_true, preds, confs)
    print(f"{split}: {len(texts)} utterances, accuracy {acc:.4f} -> results/preds_distilbert_{split}.json")


if __name__ == "__main__":
    main()
