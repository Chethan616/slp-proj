"""Turn the saved predictions into the numbers and figures the report needs.

Reads results/metrics.json and results/preds_<model>.json (written by
train_baselines.py and train_distilbert.py) and produces:

  results/model_comparison.png   accuracy / macro-F1 across the four models
  results/confusion_matrix.png   41x41 matrix for the deployed model
  results/training_curves.png    loss and validation accuracy per epoch
  results/threshold_sweep.png    out-of-scope confidence threshold analysis
  results/per_class_f1.png       the classes the deployed model finds hardest
  results/summary.md             markdown tables, ready to paste into the report
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import RESULTS, load_labels

ORDER = ["tfidf_logreg", "bow_mlp", "bilstm", "distilbert"]
DEPLOYED = "distilbert"


def load_preds(key: str):
    path = RESULTS / f"preds_{key}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.array(d["y_true"]), np.array(d["y_pred"]), np.array(d["y_conf"])


def model_comparison(metrics, present):
    names = [metrics[k]["name"] for k in present]
    acc = [metrics[k]["test_accuracy"] for k in present]
    f1 = [metrics[k]["test_macro_f1"] for k in present]

    x = np.arange(len(present))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - 0.19, acc, 0.38, label="Test accuracy", color="#6366f1")
    ax.bar(x + 0.19, f1, 0.38, label="Macro F1", color="#22c55e")
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - 0.19, a + 0.008, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + 0.19, f + 0.008, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" (", "\n(") for n in names], fontsize=9)
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("Score")
    ax.set_title("Intent classification on the CLINC150 subset (41 classes, 1500 test utterances)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=160)
    plt.close(fig)


def confusion(labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(11.5, 10))
    im = ax.imshow(cm, cmap="Blues", norm=matplotlib.colors.PowerNorm(0.5))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title("Confusion matrix — fine-tuned DistilBERT (test set)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="utterances")
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return cm


def training_curves(metrics, present):
    have = [k for k in present if metrics[k].get("history")]
    if not have:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key in have:
        hist = metrics[key]["history"]
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["train_loss"] for h in hist], label=metrics[key]["name"])
        axes[1].plot(ep, [h["val_accuracy"] for h in hist], label=metrics[key]["name"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss"); axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation accuracy"); axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "training_curves.png", dpi=160)
    plt.close(fig)


def threshold_sweep(labels, y_true, y_pred, y_conf):
    """The deployed app rejects a prediction whose softmax probability falls
    below a threshold and answers 'out of scope' instead. This finds the
    threshold that best balances keeping in-scope answers against catching
    out-of-scope questions."""
    oos_id = labels.index("oos")
    is_oos = y_true == oos_id

    thresholds = np.arange(0.0, 0.96, 0.02)
    in_acc, oos_rec, overall_f1 = [], [], []
    for t in thresholds:
        adj = np.where(y_conf < t, oos_id, y_pred)
        in_acc.append(float((adj[~is_oos] == y_true[~is_oos]).mean()))
        oos_rec.append(float((adj[is_oos] == oos_id).mean()))
        overall_f1.append(float(f1_score(y_true, adj, average="macro")))

    best_i = int(np.argmax(overall_f1))
    best_t = float(thresholds[best_i])

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(thresholds, in_acc, label="In-scope accuracy", color="#6366f1")
    ax.plot(thresholds, oos_rec, label="Out-of-scope recall", color="#f59e0b")
    ax.plot(thresholds, overall_f1, label="Overall macro F1", color="#22c55e", ls="--")
    ax.axvline(best_t, color="#ef4444", ls=":", lw=1.4,
               label=f"chosen threshold = {best_t:.2f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Out-of-scope rejection: accuracy / recall trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_sweep.png", dpi=160)
    plt.close(fig)

    return {
        "best_threshold": best_t,
        "in_scope_accuracy": in_acc[best_i],
        "oos_recall": oos_rec[best_i],
        "macro_f1": overall_f1[best_i],
        "at_zero": {
            "in_scope_accuracy": in_acc[0],
            "oos_recall": oos_rec[0],
            "macro_f1": overall_f1[0],
        },
    }


def per_class_f1(labels, y_true, y_pred):
    rep = classification_report(
        y_true, y_pred, labels=range(len(labels)), target_names=labels,
        output_dict=True, zero_division=0,
    )
    rows = sorted(
        ((name, rep[name]["f1-score"], rep[name]["support"]) for name in labels),
        key=lambda r: r[1],
    )
    worst = rows[:12]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh([r[0] for r in worst], [r[1] for r in worst], color="#f97316")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 score")
    ax.set_title("Twelve hardest intents for the deployed model")
    ax.grid(axis="x", alpha=0.25)
    for i, r in enumerate(worst):
        ax.text(r[1] + 0.012, i, f"{r[1]:.2f}", va="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "per_class_f1.png", dpi=160)
    plt.close(fig)
    return rep, rows


def main() -> None:
    labels = load_labels()
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    present = [k for k in ORDER if k in metrics]

    print(f"{'Model':<32}{'Params':>12}{'Accuracy':>11}{'Macro F1':>11}{'Train':>9}")
    print("-" * 75)
    for key in present:
        m = metrics[key]
        print(
            f"{m['name']:<32}{m['params']:>12,}{m['test_accuracy']:>11.4f}"
            f"{m['test_macro_f1']:>11.4f}{m['train_seconds']:>8.0f}s"
        )

    model_comparison(metrics, present)
    training_curves(metrics, present)

    lines = [
        "# Results",
        "",
        "## Model comparison (1500 held-out test utterances, 41 classes)",
        "",
        "| Model | Family | Trainable parameters | Test accuracy | Macro F1 | Training time |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in present:
        m = metrics[key]
        star = " **(deployed)**" if m.get("deployed") else ""
        lines.append(
            f"| {m['name']}{star} | {m['family']} | {m['params']:,} | "
            f"{m['test_accuracy']:.4f} | {m['test_macro_f1']:.4f} | {m['train_seconds']:.0f}s |"
        )

    dep = load_preds(DEPLOYED)
    if dep is not None:
        y_true, y_pred, y_conf = dep
        cm = confusion(labels, y_true, y_pred)
        sweep = threshold_sweep(labels, y_true, y_pred, y_conf)
        rep, rows = per_class_f1(labels, y_true, y_pred)

        oos_id = labels.index("oos")
        in_mask = y_true != oos_id
        print(f"\nDeployed model, split by scope:")
        print(f"  in-scope accuracy (no threshold) : {(y_pred[in_mask] == y_true[in_mask]).mean():.4f}")
        print(f"  out-of-scope recall (no threshold): {(y_pred[~in_mask] == oos_id).mean():.4f}")
        print(f"\nBest confidence threshold: {sweep['best_threshold']:.2f}")
        print(f"  in-scope accuracy : {sweep['in_scope_accuracy']:.4f}")
        print(f"  out-of-scope recall: {sweep['oos_recall']:.4f}")
        print(f"  overall macro F1   : {sweep['macro_f1']:.4f}")
        print(f"\nHardest intents: " + ", ".join(f"{n} ({f:.2f})" for n, f, _ in rows[:6]))

        confusions = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and cm[i, j] > 0:
                    confusions.append((cm[i, j], labels[i], labels[j]))
        confusions.sort(reverse=True)

        lines += [
            "",
            "## Out-of-scope handling (deployed model)",
            "",
            f"- Confidence threshold selected on macro F1: **{sweep['best_threshold']:.2f}**",
            f"- In-scope accuracy at that threshold: **{sweep['in_scope_accuracy']:.4f}**",
            f"- Out-of-scope recall at that threshold: **{sweep['oos_recall']:.4f}**",
            f"- Overall macro F1 at that threshold: **{sweep['macro_f1']:.4f}**",
            f"- Without any threshold: in-scope {sweep['at_zero']['in_scope_accuracy']:.4f}, "
            f"OOS recall {sweep['at_zero']['oos_recall']:.4f}",
            "",
            "## Most frequent confusions (deployed model)",
            "",
            "| True intent | Predicted as | Count |",
            "|---|---|---:|",
        ]
        lines += [f"| {t} | {p} | {c} |" for c, t, p in confusions[:10]]
        lines += [
            "",
            "## Hardest intents by F1 (deployed model)",
            "",
            "| Intent | F1 | Support |",
            "|---|---:|---:|",
        ]
        lines += [f"| {n} | {f:.3f} | {int(s)} |" for n, f, s in rows[:10]]

        (RESULTS / "threshold.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nFigures and summary.md written to {RESULTS}")


if __name__ == "__main__":
    main()
