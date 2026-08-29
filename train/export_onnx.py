"""Export the fine-tuned classifier to int8 ONNX for deployment.

The deployed app does not need PyTorch: faster-whisper runs on CTranslate2, and
the intent classifier runs just as well under ONNX Runtime. Dropping torch from
the runtime removes roughly 2.5 GB of CUDA libraries that the default Linux
wheel pulls in, which matters on a free hosting tier with a fixed disk and
memory budget.

This also verifies that quantisation did not change the model's predictions,
because the numbers reported in the report were measured with the PyTorch model.

Writes app/model_onnx/.
"""

import json
import time

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import ROOT, load_split, update_metrics

SRC = ROOT / "app" / "model"
OUT = ROOT / "app" / "model_onnx"
MAX_LEN = 32


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC)
    model.eval()

    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    # --- export ------------------------------------------------------------
    dummy = tok("export sample", truncation=True, padding="max_length",
                max_length=MAX_LEN, return_tensors="pt")
    fp32 = OUT / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
        # The dynamo exporter (torch >= 2.9 default) emits a graph whose shape
        # inference the ONNX Runtime quantiser rejects; the legacy TorchScript
        # exporter produces a graph it handles cleanly. Inputs are always padded
        # to MAX_LEN, so only the batch axis needs to be dynamic.
        dynamo=False,
    )
    print(f"exported fp32 ONNX: {fp32.stat().st_size / 1e6:.1f} MB")

    # --- quantise ----------------------------------------------------------
    int8 = OUT / "model.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"quantised int8 ONNX: {int8.stat().st_size / 1e6:.1f} MB")

    # The exporter writes weights above 2 GB-safe limits into a sidecar
    # `.onnx.data` file. The quantised graph is self-contained, so drop both the
    # fp32 graph and its sidecar rather than shipping 255 MB of dead weight.
    fp32.unlink(missing_ok=True)
    for leftover in OUT.glob("model_fp32.onnx*"):
        leftover.unlink()

    tok.save_pretrained(OUT)
    (OUT / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")

    # --- parity check ------------------------------------------------------
    texts, y_true = load_split("test")
    sess = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])

    onnx_pred, torch_pred = [], []
    t_onnx = t_torch = 0.0
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        enc = tok(batch, truncation=True, padding="max_length",
                  max_length=MAX_LEN, return_tensors="np")

        t0 = time.perf_counter()
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        t_onnx += time.perf_counter() - t0
        onnx_pred.extend(logits.argmax(axis=-1).tolist())

        t0 = time.perf_counter()
        with torch.no_grad():
            tl = model(
                input_ids=torch.tensor(enc["input_ids"]),
                attention_mask=torch.tensor(enc["attention_mask"]),
            ).logits
        t_torch += time.perf_counter() - t0
        torch_pred.extend(tl.argmax(dim=-1).tolist())

    agreement = float(np.mean(np.array(onnx_pred) == np.array(torch_pred)))
    onnx_acc = accuracy_score(y_true, onnx_pred)
    onnx_f1 = f1_score(y_true, onnx_pred, average="macro")
    torch_acc = accuracy_score(y_true, torch_pred)

    print(f"\nagreement with PyTorch model : {agreement:.4f}")
    print(f"PyTorch test accuracy        : {torch_acc:.4f}")
    print(f"ONNX int8 test accuracy      : {onnx_acc:.4f}  (macro F1 {onnx_f1:.4f})")
    print(f"inference time, PyTorch      : {t_torch:.2f}s")
    print(f"inference time, ONNX int8    : {t_onnx:.2f}s  ({t_torch / t_onnx:.1f}x faster)")

    update_metrics("distilbert_onnx_int8", {
        "name": "DistilBERT, ONNX int8 (deployed build)",
        "family": "transformer (quantised for deployment)",
        "size_mb": round(int8.stat().st_size / 1e6, 1),
        "test_accuracy": float(onnx_acc),
        "test_macro_f1": float(onnx_f1),
        "agreement_with_fp32": agreement,
        "speedup_vs_torch": round(t_torch / t_onnx, 2),
    })
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
