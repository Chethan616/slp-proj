"""Three reference models for the intent classifier comparison.

  1. TF-IDF + Logistic Regression   -- classical, no neural network at all
  2. Bag-of-words -> MLP            -- the textbook "chatbot neural net"
  3. Embedding -> BiLSTM -> Dense   -- classic sequence model

These exist so the report can show what the fine-tuned transformer actually buys
you. None of them is deployed. Everything is PyTorch (plus scikit-learn for the
classical baseline); TensorFlow is deliberately not a dependency of this project.

Writes results/metrics.json entries and results/preds_<model>.json.
"""

import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from common import (
    count_params,
    load_labels,
    load_split,
    save_predictions,
    set_seed,
    update_metrics,
)

DEVICE = torch.device("cpu")
MAX_LEN = 32
TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def evaluate(logits: torch.Tensor, y_true: list[int]) -> tuple[float, float, np.ndarray, np.ndarray]:
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    pred = pred.cpu().numpy()
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro")
    return acc, f1, pred, conf.cpu().numpy()


# --------------------------------------------------------------------------
# 1. TF-IDF + Logistic Regression
# --------------------------------------------------------------------------
def run_tfidf_logreg(train, val, test, n_classes):
    print("\n[1/3] TF-IDF + Logistic Regression")
    (Xtr, ytr), (_, _), (Xte, yte) = train, val, test

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=2000, C=10.0, n_jobs=-1)
    clf.fit(Xtr_v, ytr)
    train_s = time.perf_counter() - t0

    probs = clf.predict_proba(Xte_v)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")

    print(f"      vocab={len(vec.vocabulary_)}  test acc={acc:.4f}  macro-F1={f1:.4f}")
    save_predictions("tfidf_logreg", yte, pred, conf)
    update_metrics(
        "tfidf_logreg",
        {
            "name": "TF-IDF + Logistic Regression",
            "family": "classical",
            "params": int(clf.coef_.size + clf.intercept_.size),
            "features": len(vec.vocabulary_),
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1),
            "train_seconds": round(train_s, 1),
        },
    )


# --------------------------------------------------------------------------
# 2. Bag-of-words -> MLP
# --------------------------------------------------------------------------
class BowMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_bow_mlp(train, val, test, n_classes):
    print("\n[2/3] Bag-of-words -> MLP")
    set_seed()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train, val, test

    vec = CountVectorizer(binary=True, min_df=2, tokenizer=tokenize, token_pattern=None)
    Xtr_v = torch.tensor(vec.fit_transform(Xtr).toarray(), dtype=torch.float32)
    Xva_v = torch.tensor(vec.transform(Xva).toarray(), dtype=torch.float32)
    Xte_v = torch.tensor(vec.transform(Xte).toarray(), dtype=torch.float32)
    ytr_t = torch.tensor(ytr)
    yva_t = torch.tensor(yva)

    model = BowMLP(Xtr_v.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    history = []

    t0 = time.perf_counter()
    best_f1, best_state = 0.0, None
    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(len(Xtr_v))
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr_v[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_v)
            val_loss = lossf(val_logits, yva_t).item()
            val_acc, val_f1, _, _ = evaluate(val_logits, yva)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(perm),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(f"      epoch {epoch:>2}  train_loss={total/len(perm):.4f}  val_acc={val_acc:.4f}")

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc, f1, pred, conf = evaluate(model(Xte_v), yte)

    print(f"      vocab={Xtr_v.shape[1]}  test acc={acc:.4f}  macro-F1={f1:.4f}")
    save_predictions("bow_mlp", yte, pred, conf)
    update_metrics(
        "bow_mlp",
        {
            "name": "Bag-of-words + MLP",
            "family": "neural (from scratch)",
            "params": count_params(model),
            "features": int(Xtr_v.shape[1]),
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1),
            "train_seconds": round(train_s, 1),
            "history": history,
        },
    )


# --------------------------------------------------------------------------
# 3. Embedding -> BiLSTM -> Dense
# --------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab: int, n_classes: int, emb: int = 128, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        out, _ = self.lstm(self.emb(x))
        # Masked mean-pool over real tokens only, so padding cannot dilute the
        # sentence representation.
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(self.drop(pooled))


def encode(texts, vocab):
    out = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(t, 1) for t in tokenize(text)][:MAX_LEN]
        out[i, : len(ids)] = ids
    return torch.tensor(out)


def run_bilstm(train, val, test, n_classes):
    print("\n[3/3] Embedding -> BiLSTM -> Dense")
    set_seed()
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train, val, test

    counts = Counter(t for text in Xtr for t in tokenize(text))
    # 0 = padding, 1 = unknown
    vocab = {tok: i + 2 for i, (tok, c) in enumerate(counts.most_common()) if c >= 2}
    vocab_size = len(vocab) + 2

    Xtr_v, Xva_v, Xte_v = encode(Xtr, vocab), encode(Xva, vocab), encode(Xte, vocab)
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)

    model = BiLSTMClassifier(vocab_size, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    history = []

    t0 = time.perf_counter()
    best_f1, best_state = 0.0, None
    for epoch in range(1, 41):
        model.train()
        perm = torch.randperm(len(Xtr_v))
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr_v[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_v)
            val_loss = lossf(val_logits, yva_t).item()
            val_acc, val_f1, _, _ = evaluate(val_logits, yva)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(perm),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0:
            print(f"      epoch {epoch:>2}  train_loss={total/len(perm):.4f}  val_acc={val_acc:.4f}")

    train_s = time.perf_counter() - t0
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        acc, f1, pred, conf = evaluate(model(Xte_v), yte)

    print(f"      vocab={vocab_size}  test acc={acc:.4f}  macro-F1={f1:.4f}")
    save_predictions("bilstm", yte, pred, conf)
    update_metrics(
        "bilstm",
        {
            "name": "Embedding + BiLSTM",
            "family": "neural (from scratch)",
            "params": count_params(model),
            "features": vocab_size,
            "test_accuracy": float(acc),
            "test_macro_f1": float(f1),
            "train_seconds": round(train_s, 1),
            "history": history,
        },
    )


def main() -> None:
    set_seed()
    labels = load_labels()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    print(f"train={len(train[0])}  val={len(val[0])}  test={len(test[0])}  classes={len(labels)}")

    run_tfidf_logreg(train, val, test, len(labels))
    run_bow_mlp(train, val, test, len(labels))
    run_bilstm(train, val, test, len(labels))
    print("\nBaselines done -> results/metrics.json")


if __name__ == "__main__":
    main()
