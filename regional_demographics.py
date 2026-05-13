"""
Regional demographic sampling and validation.

The latent variable is native_state. Religion, caste, mother tongue, and name
style come from the native profile. Current state/city may differ through a
15-25% migration process, which creates realistic cases such as a Tamil speaker
living in Bangalore without destroying the original linguistic covariance.
"""

import random

from constants import (
    ALL_CITIES,
    CASTES_BY_RELIGION,
    CASTES_BY_STATE_AND_RELIGION,
    CITY_STATE_MAP,
    CITY_WEIGHTS,
    MIGRANT_MOTHER_TONGUE_DIST,
    MIGRATION_PROBABILITY_RANGE,
    REGIONAL_PRIORS,
    TIER_1_CITIES,
    TIER_2_CITIES,
)
from name_generator import generate_name, infer_name_style


MIGRATION_DESTINATION_BONUS = {
    "Maharashtra": 1.45,
    "Delhi": 1.45,
    "Karnataka": 1.35,
    "Telangana": 1.30,
    "Tamil Nadu": 1.15,
    "Gujarat": 1.15,
}


def weighted_pick(dist):
    keys = list(dist.keys())
    weights = list(dist.values())
    return random.choices(keys, weights=weights, k=1)[0]


def normalize(dist):
    total = sum(dist.values())
    return {key: value / total for key, value in dist.items() if value > 0}


def blend_distributions(primary_dist, secondary_dist, secondary_weight):
    keys = set(primary_dist) | set(secondary_dist)
    primary_weight = 1 - secondary_weight
    blended = {
        key: (
            primary_dist.get(key, 0) * primary_weight
            + secondary_dist.get(key, 0) * secondary_weight
        )
        for key in keys
    }
    return normalize(blended)


def pick_native_state():
    return weighted_pick({
        state: prior["state_weight"]
        for state, prior in REGIONAL_PRIORS.items()
    })


def city_distribution_for_state(state):
    return {
        city: CITY_WEIGHTS[idx]
        for idx, city in enumerate(ALL_CITIES)
        if CITY_STATE_MAP[city] == state
    }


def pick_city_for_state(state):
    return weighted_pick(city_distribution_for_state(state))


def pick_migration_probability():
    low, high = MIGRATION_PROBABILITY_RANGE
    return random.uniform(low, high)


def migration_target_distribution(native_state):
    dist = {}
    for state, prior in REGIONAL_PRIORS.items():
        if state == native_state:
            continue
        dist[state] = prior["state_weight"] * MIGRATION_DESTINATION_BONUS.get(state, 1.0)
    return normalize(dist)


def generate_city(native_state, migration=True, migration_probability=None):
    """
    Generate current city/state.

    Migration is low-probability and weighted toward common employment hubs,
    but it is never deterministic. Non-migrants stay in their native state.
    """
    probability = (
        pick_migration_probability()
        if migration_probability is None
        else migration_probability
    )
    migrated = migration and random.random() < probability
    current_state = (
        weighted_pick(migration_target_distribution(native_state))
        if migrated else native_state
    )
    return {
        "state": current_state,
        "city": pick_city_for_state(current_state),
        "migrated": migrated,
    }


def migration_noise_for_city(city, migrated):
    if migrated:
        return 0.07 if city in TIER_1_CITIES else 0.05
    if city in TIER_1_CITIES:
        return 0.18
    if city in TIER_2_CITIES:
        return 0.12
    return 0.08


def pick_mother_tongue(native_state, current_city, migrated):
    """
    Preserve native linguistic profile with small multilingual/urban noise.

    For migrants, the native language still dominates; the current city adds
    only a small chance of Hindi/English/other household languages.
    """
    native_dist = REGIONAL_PRIORS[native_state]["mother_tongues"]
    tongue_dist = blend_distributions(
        native_dist,
        MIGRANT_MOTHER_TONGUE_DIST,
        migration_noise_for_city(current_city, migrated),
    )
    return weighted_pick(tongue_dist)


def pick_caste(native_state, religion):
    state_castes = CASTES_BY_STATE_AND_RELIGION.get(native_state, {})
    caste_dist = state_castes.get(religion)
    if caste_dist:
        return weighted_pick(caste_dist)

    fallback = CASTES_BY_RELIGION.get(religion)
    if fallback:
        return random.choice(fallback)
    return "Not specified"


def generate_regional_demographics(gender, migration=True, migration_probability=None):
    """
    Generate state -> city -> religion -> caste -> mother_tongue -> name.

    This function uses Python's global random module, so random.seed(...) remains
    the deterministic seeding hook for demographic sampling.
    """
    native_state = pick_native_state()
    native_profile = REGIONAL_PRIORS[native_state]

    religion = weighted_pick(native_profile["religions"])
    caste = pick_caste(native_state, religion)

    current_location = generate_city(
        native_state,
        migration=migration,
        migration_probability=migration_probability,
    )
    mother_tongue = pick_mother_tongue(
        native_state,
        current_location["city"],
        current_location["migrated"],
    )

    name_style = native_profile["name_style"]
    name = generate_name(
        gender=gender,
        religion=religion,
        style=name_style,
        caste=caste,
    )

    return {
        "native_state": native_state,
        "state": current_location["state"],
        "city": current_location["city"],
        "migrated": current_location["migrated"],
        "religion": religion,
        "caste": caste,
        "mother_tongue": mother_tongue,
        "name_style": name_style,
        "name": name,
    }


def caste_is_possible(religion, caste):
    if caste == "Not specified":
        return True
    valid = set(CASTES_BY_RELIGION.get(religion, []))
    return caste in valid


def validate_profile_demographic_consistency(profile):
    """
    Return a soft demographic consistency score in [0, 1].

    Low scores indicate combinations that are possible but statistically odd,
    or impossible caste/religion pairings. This intentionally avoids binary
    pass/fail logic because legitimate users can be multilingual, migrated,
    intercultural, or anglicized.
    """
    score = 1.0

    state = profile.get("native_state") or profile.get("state")
    current_state = profile.get("state")
    religion = profile.get("religion")
    caste = profile.get("caste")
    mother_tongue = profile.get("mother_tongue")
    name = profile.get("name", "")

    native_profile = REGIONAL_PRIORS.get(state)
    current_profile = REGIONAL_PRIORS.get(current_state)
    if not native_profile:
        return 0.45

    if religion not in native_profile["religions"]:
        score -= 0.25
    elif native_profile["religions"][religion] < 0.01:
        score -= 0.12

    if not caste_is_possible(religion, caste):
        score -= 0.35

    native_tongue_prob = native_profile["mother_tongues"].get(mother_tongue, 0)
    current_tongue_prob = (
        current_profile["mother_tongues"].get(mother_tongue, 0)
        if current_profile else 0
    )
    migrant_prob = MIGRANT_MOTHER_TONGUE_DIST.get(mother_tongue, 0)
    if max(native_tongue_prob, current_tongue_prob, migrant_prob) == 0:
        score -= 0.25
    elif max(native_tongue_prob, current_tongue_prob, migrant_prob) < 0.03:
        score -= 0.10

    expected_style = native_profile["name_style"]
    inferred_style = infer_name_style(name)
    if inferred_style is None:
        score -= 0.08
    elif inferred_style != expected_style:
        score -= 0.18

    return round(max(0.0, min(1.0, score)), 3)
