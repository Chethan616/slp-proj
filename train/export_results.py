"""Export the measured results as JSON for the web interface.

The site shows the same numbers as the report, and both read from results/, so
neither can quietly drift from the experiments. Run after evaluate.py,
export_onnx.py and eval_voice.py.

Writes web/src/results.json.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from common import RESULTS, ROOT, load_labels

OUT = ROOT / "web" / "src" / "results.json"

MODEL_ORDER = ["tfidf_logreg", "bow_mlp", "bilstm", "distilbert"]
SHORT = {
    "tfidf_logreg": "TF-IDF + LogReg",
    "bow_mlp": "Bag-of-words + MLP",
    "bilstm": "Embedding + BiLSTM",
    "distilbert": "DistilBERT",
}


def main() -> None:
    labels = load_labels()
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    thresh = json.loads((RESULTS / "threshold.json").read_text(encoding="utf-8"))
    voice = json.loads((RESULTS / "voice_eval.json").read_text(encoding="utf-8"))["summary"]
    preds = json.loads((RESULTS / "preds_distilbert.json").read_text(encoding="utf-8"))

    y_true = np.array(preds["y_true"])
    y_pred = np.array(preds["y_pred"])

    rep = classification_report(
        y_true, y_pred, labels=range(len(labels)), target_names=labels,
        output_dict=True, zero_division=0,
    )
    per_class = sorted(
        ({"intent": n, "f1": round(rep[n]["f1-score"], 4),
          "precision": round(rep[n]["precision"], 4),
          "recall": round(rep[n]["recall"], 4),
          "support": int(rep[n]["support"])} for n in labels),
        key=lambda r: r["f1"],
    )

    # The confusions worth naming, for the table under the per-class chart.
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    confusions = sorted(
        ((int(cm[i, j]), labels[i], labels[j])
         for i in range(len(labels)) for j in range(len(labels))
         if i != j and cm[i, j] > 0),
        reverse=True,
    )[:6]

    splits = {
        s: len(json.loads((ROOT / "data" / f"{s}.json").read_text(encoding="utf-8")))
        for s in ("train", "val", "test")
    }

    onnx = metrics.get("distilbert_onnx_int8", {})
    dep = metrics.get("distilbert", {})

    out = {
        "classes": len(labels),
        "intents": labels,
        "splits": splits,
        "headline": {
            "accuracy": onnx.get("test_accuracy", dep.get("test_accuracy")),
            "macro_f1": onnx.get("test_macro_f1", dep.get("test_macro_f1")),
            "wer": voice["corpus_wer"],
            "oos_recall": thresh["test"]["oos_recall"],
        },
        "models": [
            {
                "key": k,
                "name": SHORT[k],
                "params": metrics[k]["params"],
                "accuracy": metrics[k]["test_accuracy"],
                "macro_f1": metrics[k]["test_macro_f1"],
                "family": metrics[k]["family"],
            }
            for k in MODEL_ORDER if k in metrics
        ],
        "deployed": {
            "name": "DistilBERT, int8 ONNX",
            "size_mb": onnx.get("size_mb"),
            "accuracy": onnx.get("test_accuracy"),
            "macro_f1": onnx.get("test_macro_f1"),
            "agreement_with_fp32": onnx.get("agreement_with_fp32"),
            "speedup": onnx.get("speedup_vs_torch"),
        },
        "threshold": thresh,
        "voice": voice,
        "per_class": per_class,
        "confusions": [{"count": c, "true": t, "predicted": p} for c, t, p in confusions],
        "training": dep.get("history", []),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  {out['classes']} classes, {len(out['models'])} models, "
          f"{len(out['per_class'])} per-class rows, {len(out['training'])} epochs")


if __name__ == "__main__":
    main()
