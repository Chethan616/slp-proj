"""Fine-tune DistilBERT for 41-way intent classification. This is the model
that actually ships in the deployed app.

DistilBERT is a 6-layer distillation of BERT-base: ~40% smaller and ~60% faster
at inference while retaining most of the accuracy, which is what makes it viable
on a 2-vCPU free-tier container.

Saves the best checkpoint (by validation macro-F1) to app/model/.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import (
    ROOT,
    count_params,
    load_labels,
    load_split,
    save_predictions,
    set_seed,
    update_metrics,
)

CHECKPOINT = "distilbert-base-uncased"
OUT_DIR = ROOT / "app" / "model"
MAX_LEN = 32
BATCH = 32
EPOCHS = 5
LR = 3e-5
DEVICE = torch.device("cpu")


def make_loader(tok, texts, labels, shuffle: bool) -> DataLoader:
    enc = tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_probs, all_true = [], []
    for input_ids, attn, y in loader:
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        all_probs.append(torch.softmax(logits, dim=-1))
        all_true.append(y)
    probs = torch.cat(all_probs)
    y_true = torch.cat(all_true).numpy()
    conf, pred = probs.max(dim=-1)
    return y_true, pred.numpy(), conf.numpy(), probs.numpy()


def main() -> None:
    set_seed()
    labels = load_labels()
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}

    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    print(f"train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}  classes={len(labels)}")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)
    print(f"trainable parameters: {count_params(model):,}")

    train_loader = make_loader(tok, Xtr, ytr, shuffle=True)
    val_loader = make_loader(tok, Xva, yva, shuffle=False)
    test_loader = make_loader(tok, Xte, yte, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    lossf = nn.CrossEntropyLoss()

    history = []
    best_f1, best_state = 0.0, None
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        ep_t0 = time.perf_counter()
        for step, (input_ids, attn, y) in enumerate(train_loader, 1):
            opt.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(y)
            seen += len(y)
            if step % 40 == 0:
                print(f"      epoch {epoch} step {step}/{len(train_loader)}  loss={running/seen:.4f}")

        y_true, pred, _, _ = predict(model, val_loader)
        val_acc = accuracy_score(y_true, pred)
        val_f1 = f1_score(y_true, pred, average="macro")
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / seen,
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1),
            }
        )
        print(
            f"  epoch {epoch}/{EPOCHS}  train_loss={running/seen:.4f}  "
            f"val_acc={val_acc:.4f}  val_macro_f1={val_f1:.4f}  "
            f"({time.perf_counter()-ep_t0:.0f}s)"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"      new best (val macro-F1 {val_f1:.4f})")

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)

    y_true, pred, conf, _ = predict(model, test_loader)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro")
    print(f"\nTEST  accuracy={acc:.4f}  macro-F1={f1:.4f}  ({train_s/60:.1f} min training)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"saved -> {OUT_DIR}")

    save_predictions("distilbert", y_true, pred, conf)
    update_metrics(
        "distilbert",
        {
            "name": "DistilBERT (fine-tuned)",
            "family": "transformer (pre-trained)",
            "params": count_params(model),
            "features": tok.vocab_size,
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1),
            "train_seconds": round(train_s, 1),
            "hyperparameters": {
                "checkpoint": CHECKPOINT,
                "max_len": MAX_LEN,
                "batch_size": BATCH,
                "epochs": EPOCHS,
                "lr": LR,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
            },
            "history": history,
            "deployed": True,
        },
    )


if __name__ == "__main__":
    main()
