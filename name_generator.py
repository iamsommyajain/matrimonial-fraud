"""
Regional name generation helpers.

Names are sampled probabilistically from regional style pools. The helpers keep
religion and caste signals correlated with names without turning them into hard
rules: low-probability anglicized names and generic regional surnames remain
possible so legitimate profiles do not become perfectly clean artifacts.
"""

import random

from constants import NAMES_BY_STYLE


ANGLICIZED_FIRST_NAMES = {
    "Male": ["Neil", "Ryan", "Kevin", "Aaron", "Vivian"],
    "Female": ["Sara", "Riya", "Tina", "Rachel", "Anita"],
}


def _pick(pool, weights=None):
    if weights:
        return random.choices(pool, weights=weights, k=1)[0]
    return random.choice(pool)


def _name_pool_for_profile(style, religion):
    style_pool = NAMES_BY_STYLE[style]
    return style_pool.get(religion, style_pool)


def _pick_hindu_surname(style_pool, caste):
    caste_surnames = style_pool.get("HinduSurnamesByCaste", {})
    if caste in caste_surnames and random.random() < 0.82:
        return _pick(caste_surnames[caste])
    return _pick(style_pool["Surnames"])


def _pick_surname(gender, religion, style, caste):
    style_pool = NAMES_BY_STYLE[style]

    if religion == "Sikh":
        return "Singh" if gender == "Male" else "Kaur"

    if religion == "Hindu":
        return _pick_hindu_surname(style_pool, caste)

    religion_pool = style_pool.get(religion)
    if religion_pool:
        return _pick(religion_pool["Surnames"])

    return _pick(style_pool["Surnames"])


def generate_name(gender, religion, style, caste, anglicized_probability=0.04):
    """
    Generate first_name + surname for a regional style.

    Seeding compatibility: this uses Python's global random module, matching the
    existing generator. Callers that seed random.seed(...) keep deterministic
    replay behavior for this layer.
    """
    pool = _name_pool_for_profile(style, religion)
    if random.random() < anglicized_probability:
        first_name = _pick(ANGLICIZED_FIRST_NAMES[gender])
    else:
        first_name = _pick(pool[gender])

    surname = _pick_surname(gender, religion, style, caste)
    return f"{first_name} {surname}"


def infer_name_style(name):
    """
    Return the most likely style for a generated-style name, or None.

    This is intentionally soft: names can overlap across regions, and validation
    should penalize uncertainty rather than treat it as proof of fraud.
    """
    if not name:
        return None

    parts = name.split()
    first = parts[0]
    surname = parts[-1] if len(parts) > 1 else ""
    scores = {}

    for style, style_pool in NAMES_BY_STYLE.items():
        score = 0
        for key, value in style_pool.items():
            if key in ("Male", "Female"):
                if first in value:
                    score += 2
            elif key == "Surnames":
                if surname in value:
                    score += 1
            elif isinstance(value, dict):
                if first in value.get("Male", []) or first in value.get("Female", []):
                    score += 2
                if surname in value.get("Surnames", []):
                    score += 1
                for caste_surnames in value.values():
                    if isinstance(caste_surnames, list) and surname in caste_surnames:
                        score += 1

        if score:
            scores[style] = score

    if not scores:
        return None
    return max(scores, key=scores.get)
