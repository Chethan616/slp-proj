"""Condense the recipe corpus into the lookup table the app ships.

The published CSV is 450 MB of nested JSON, most of which the assistant never
reads. This keeps only the fields the food intents answer with - name, servings,
calories, the three macronutrients, diet and allergen labels, and the ingredient
names - and writes them gzipped, which is small enough to deploy.

Source: datahiveai/recipes-with-nutrition (39,447 recipes).

Writes app/recipes.json.gz.
"""

import gzip
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "recipes.json.gz"

# The three macronutrients, by the nutrient codes the corpus uses.
MACROS = {"PROCNT": "protein_g", "CHOCDF": "carbs_g", "FAT": "fat_g"}

MAX_INGREDIENTS = 12


def first_or_none(raw):
    """diet_labels and friends are JSON-encoded arrays in a CSV cell."""
    try:
        vals = json.loads(raw) if isinstance(raw, str) else []
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(v) for v in vals]


def main() -> None:
    print("downloading the recipe corpus ...")
    csv = hf_hub_download(
        "datahiveai/recipes-with-nutrition",
        "recipes-with-nutrition.csv",
        repo_type="dataset",
    )

    cols = [
        "recipe_name", "servings", "calories", "diet_labels", "health_labels",
        "cautions", "cuisine_type", "dish_type", "ingredients", "total_nutrients",
    ]
    df = pd.read_csv(csv, usecols=cols)
    print(f"  {len(df):,} recipes")

    out = []
    for row in df.itertuples(index=False):
        name = str(row.recipe_name).strip()
        if not name or name.lower() == "nan":
            continue

        try:
            servings = max(1, int(float(row.servings)))
            total_cal = float(row.calories)
        except (TypeError, ValueError):
            continue
        if not (0 < total_cal < 100_000):
            continue

        # Everything the assistant quotes is per serving; the corpus totals the
        # whole dish.
        rec = {
            "name": name,
            "servings": servings,
            "calories": round(total_cal / servings),
        }

        try:
            nutrients = json.loads(row.total_nutrients)
        except (json.JSONDecodeError, TypeError):
            nutrients = {}
        for code, key in MACROS.items():
            q = nutrients.get(code, {}).get("quantity")
            if isinstance(q, (int, float)):
                rec[key] = round(q / servings)

        try:
            ings = json.loads(row.ingredients)
            names = []
            for i in ings:
                food = str(i.get("food", "")).strip().lower()
                if food and food not in names:
                    names.append(food)
            rec["ingredients"] = names[:MAX_INGREDIENTS]
        except (json.JSONDecodeError, TypeError, AttributeError):
            rec["ingredients"] = []

        diet = first_or_none(row.diet_labels) + first_or_none(row.health_labels)
        # Keep the labels a person would actually say out loud.
        keep = {"Vegan", "Vegetarian", "Gluten-Free", "Dairy-Free", "Low-Carb",
                "High-Fiber", "High-Protein", "Balanced", "Low-Fat", "Keto-Friendly"}
        rec["diet"] = [d for d in diet if d in keep][:3]
        rec["cautions"] = first_or_none(row.cautions)[:3]
        rec["cuisine"] = (first_or_none(row.cuisine_type) or [""])[0]
        rec["dish_type"] = (first_or_none(row.dish_type) or [""])[0]

        out.append(rec)

    print(f"  kept {len(out):,} usable recipes")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))

    mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({mb:.1f} MB gzipped)")

    sample = next(r for r in out if r.get("protein_g"))
    print("\nexample record:")
    print(" ", json.dumps(sample)[:300])


if __name__ == "__main__":
    main()
