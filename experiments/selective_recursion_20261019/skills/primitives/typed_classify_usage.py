"""Typed category decisions for items (E74: a System One style leaf supplied by the harness).

CATEGORIES                     # the item categories this family's questions ask about (a list of lowercase words)
classify(item) -> str          # the one category in CATEGORIES that the item belongs to, asked as a single choice; cached
classify_many(items) -> dict   # {item: category} for every distinct item, asked in parallel; use this for many records
Typical use: cats = classify_many([r['item'] for r in records]); total = sum(r['qty'] for r in records if r['action'] == 'bought' and cats[r['item']] == category)
"""

import json
import re

# Induced without labels (E73): the helper named the category asked about in each of the 30 unlabelled teaching
# questions of every semantic-total family; the union was the same five words in all 32 families.
CATEGORIES = ["fruit", "furniture", "instrument", "tool", "vegetable"]
_CACHE = {}


def _prompt(item):
    return (f"Which category does the item '{item}' belong to? Options: {', '.join(CATEGORIES)}. "
            "Answer with one word from the options.")


def _parse(text):
    t = (text or "").strip().lower()
    return next((c for c in CATEGORIES if re.search(rf"\b{c}\b", t)), None)


def classify(item):
    key = str(item).strip().lower()
    if key not in _CACHE:
        _CACHE[key] = _parse(ask(_prompt(key)))
    return _CACHE[key]


def classify_many(items):
    keys = sorted({str(i).strip().lower() for i in items} - set(_CACHE))
    if keys:
        for k, text in zip(keys, ask_batch([_prompt(k) for k in keys]).texts):
            _CACHE[k] = _parse(text)
    return {str(i).strip().lower(): _CACHE[str(i).strip().lower()] for i in items}


json  # noqa: B018 (keep the import available to candidates that expect it)
