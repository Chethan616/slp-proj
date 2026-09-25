"""Build the food-and-nutrition subset of CLINC150 for the voice chatbot.

CLINC150 (Larson et al., EMNLP 2019) ships 150 in-scope intents across 10
domains plus a deliberate out-of-scope class. This project uses one domain in
full - kitchen and dining, 15 intents - so the assistant has a coherent subject
it can actually answer about, and keeps `oos` so it can refuse everything else
rather than guessing.

A narrow assistant is asked out-of-scope questions far more often than a general
one, which makes the out-of-scope class the most important one here.

Writes data/{train,val,test}.json and data/labels.json.
"""

import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 42

# 40 in-scope intents, grouped by CLINC domain. Chosen for voice-assistant
# realism and demo value -- every one of these is something you would plausibly
# say out loud to a phone.
SELECTED = {
    # CLINC150's kitchen-and-dining domain in full: the 15 intents a food and
    # nutrition assistant actually needs, grouped here by what they are about so
    # the interface can show which part of the domain answered.
    "cooking": [
        "recipe", "ingredients_list", "ingredient_substitution",
        "cook_time", "meal_suggestion",
    ],
    "nutrition": [
        "calories", "nutrition_info", "food_last",
    ],
    "restaurants": [
        "restaurant_suggestion", "restaurant_reviews", "how_busy",
        "restaurant_reservation", "confirm_reservation", "cancel_reservation",
        "accept_reservations",
    ],
}

# How many out-of-scope examples to keep per split. CLINC's `plus` config has
# 1000 OOS test utterances, which would swamp 40 in-scope classes (30 each), so
# the test set is subsampled to keep the confusion matrix readable while still
# leaving enough OOS to estimate recall.
OOS_KEEP = {"train": 250, "validation": 75, "test": 150}


def main() -> None:
    in_scope = [name for names in SELECTED.values() for name in names]
    assert len(in_scope) == 15, f"expected 15 intents, got {len(in_scope)}"
    assert len(set(in_scope)) == 15, "duplicate intent in SELECTED"

    print("Loading clinc/clinc_oos (config: plus) ...")
    ds = load_dataset("clinc/clinc_oos", "plus")

    # The dataset card calls the target column "label"; the actual feature name
    # has historically been "intent". Detect it rather than guessing.
    cols = ds["train"].column_names
    label_col = "intent" if "intent" in cols else "label"
    print(f"  columns={cols}  -> using label column '{label_col}'")

    names = ds["train"].features[label_col].names
    missing = sorted(set(in_scope) - set(names))
    if missing:
        raise SystemExit(
            f"These intent names are not in CLINC150: {missing}\n"
            f"Available (first 40): {names[:40]}"
        )

    # Stable label ordering: in-scope alphabetical, then oos last.
    labels = sorted(in_scope) + ["oos"]
    label2id = {name: i for i, name in enumerate(labels)}
    keep_ids = {names.index(n) for n in in_scope}
    oos_id = names.index("oos")

    rng = random.Random(SEED)
    out_counts = {}

    for split, out_name in [("train", "train"), ("validation", "val"), ("test", "test")]:
        rows = []
        oos_pool = []
        for text, lid in zip(ds[split]["text"], ds[split][label_col]):
            if lid in keep_ids:
                name = names[lid]
                rows.append({"text": text, "label": name, "label_id": label2id[name]})
            elif lid == oos_id:
                oos_pool.append(text)

        rng.shuffle(oos_pool)
        for text in oos_pool[: OOS_KEEP[split]]:
            rows.append({"text": text, "label": "oos", "label_id": label2id["oos"]})

        rng.shuffle(rows)
        path = DATA / f"{out_name}.json"
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")

        counts = Counter(r["label"] for r in rows)
        out_counts[out_name] = len(rows)
        print(
            f"  {out_name:<5} {len(rows):>5} rows  "
            f"({len(counts) - 1} in-scope classes, "
            f"{counts['oos']} oos, "
            f"{min(v for k, v in counts.items() if k != 'oos')}-"
            f"{max(v for k, v in counts.items() if k != 'oos')} per in-scope class)"
        )

    (DATA / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(f"\nWrote {len(labels)} labels to data/labels.json")
    print(f"Totals: {out_counts}")


if __name__ == "__main__":
    main()
