"""Dish lookup over the recipe corpus.

The intent model decides *what was asked*; this decides *what it was asked
about*. Together they turn "how many calories are in chicken biryani" into a
real answer rather than a fixed sentence.

Matching is deliberately simple: an inverted index over recipe-name tokens,
scored by how much of the name the query covers. There is no slot-filling model
here, and none is claimed - the query is matched against 39,447 names directly.
"""

import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "recipes.json.gz"

# Words that carry the intent rather than the dish. Stripping them stops
# "calories" matching a recipe that happens to have it in the title.
STOP = {
    "a", "an", "the", "for", "of", "in", "on", "to", "is", "are", "and", "with",
    "how", "many", "much", "what", "whats", "which", "do", "does", "i", "me",
    "my", "you", "can", "could", "would", "tell", "give", "show", "find", "get",
    "make", "making", "cook", "cooking", "there", "some", "any", "it", "its",
    "calories", "calorie", "nutrition", "nutritional", "nutrients", "recipe",
    "recipes", "ingredient", "ingredients", "need", "needed", "long", "take",
    "takes", "time", "serving", "servings", "per", "about", "have", "has",
    "want", "like", "please", "know", "info", "information", "dish", "food",
    "meal", "eat", "eating", "protein", "carbs", "fat", "healthy", "good",
}

TOKEN = re.compile(r"[a-z]+")

_recipes = None
_index = None
_idf = None


def _tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 2]


def _load():
    """Build the inverted index once, on first lookup."""
    global _recipes, _index, _idf
    if _recipes is not None:
        return

    with gzip.open(DATA, "rt", encoding="utf-8") as fh:
        _recipes = json.load(fh)

    _index = defaultdict(list)
    doc_freq = defaultdict(int)
    for i, r in enumerate(_recipes):
        seen = set(_tokens(r["name"]))
        for t in seen:
            _index[t].append(i)
            doc_freq[t] += 1

    n = len(_recipes)
    # Rare words identify a dish; common ones ("chicken") barely narrow it.
    _idf = {t: math.log(n / (1 + df)) for t, df in doc_freq.items()}


def lookup(text):
    """Best-matching recipe for a query, or None when nothing matches well."""
    _load()

    query = _tokens(text)
    if not query:
        return None

    scores = defaultdict(float)
    hits = defaultdict(int)
    for t in set(query):
        if t not in _index:
            continue
        w = _idf.get(t, 0.0)
        for i in _index[t]:
            scores[i] += w
            hits[i] += 1

    if not scores:
        return None

    # Normalise by name length so a short, exact title beats a long one that
    # merely happens to contain the same words.
    ranked = max(
        scores.items(),
        key=lambda kv: kv[1] / math.sqrt(len(_tokens(_recipes[kv[0]]["name"])) or 1),
    )
    best_i = ranked[0]
    best = scores[best_i]

    # A query of several words has to match on several words. Without this,
    # "set an alarm" matches any title sharing one incidental token.
    needed = 2 if len(set(query)) >= 2 else 1
    if hits[best_i] < needed:
        return None

    query_weight = sum(_idf.get(t, 0.0) for t in set(query))
    coverage = best / max(1e-6, query_weight)
    if coverage < 0.45:
        return None

    rec = dict(_recipes[best_i])
    rec["_score"] = round(best, 2)
    rec["_coverage"] = round(coverage, 2)
    rec["_matched_tokens"] = hits[best_i]
    return rec


def describe(rec):
    """The placeholder values a response template can use."""
    if not rec:
        return {}

    ings = rec.get("ingredients") or []
    out = {
        "dish": rec["name"],
        "servings": str(rec["servings"]),
        "calories": str(rec["calories"]),
        "ingredient_count": str(len(ings)),
        "ingredients": ", ".join(ings[:8]) if ings else "",
        "diet_labels": ", ".join(rec.get("diet") or []).lower() or "unlabelled",
    }
    for key, label, unit in [("protein_g", "protein", "g"),
                             ("carbs_g", "carbs", "g"),
                             ("fat_g", "fat", "g")]:
        if rec.get(key) is not None:
            out[label] = f"{rec[key]} {unit}"
    return out


def count():
    _load()
    return len(_recipes)
