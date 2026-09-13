"""
Demographically conditioned hobby generation.

Uniform random hobbies create synthetic artifacts. Real matrimonial profiles are
repetitive, long-tailed, and clustered: music/movies/cooking/travel dominate,
while niche interests appear rarely and usually co-occur with related hobbies.
This module models that covariance so downstream fraud/inconsistency models can
learn lifestyle coherence instead of memorizing flat hobby lists.
"""

import random

from .constants import (
    CITY_METADATA,
    COMMON_HOBBIES,
    HOBBY_CLUSTERS,
    HOBBY_COOCCURRENCE_GROUPS,
    HOBBY_INCOMPATIBLE_GROUPS,
    HOBBY_TIER_WEIGHTS,
    MODERATELY_COMMON_HOBBIES,
    NICHE_HOBBIES,
    RARE_HOBBIES,
    TIER_1_CITIES,
    TIER_2_CITIES,
)


HOBBY_TIERS = {
    "common": COMMON_HOBBIES,
    "moderate": MODERATELY_COMMON_HOBBIES,
    "niche": NICHE_HOBBIES,
    "rare": RARE_HOBBIES,
}


def _weighted_pick(dist):
    keys = list(dist.keys())
    weights = list(dist.values())
    return random.choices(keys, weights=weights, k=1)[0]


def _normalize(dist):
    total = sum(dist.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in dist.items() if value > 0}


def _city_tier(city):
    if city in CITY_METADATA:
        return CITY_METADATA[city]["tier"]
    if city in TIER_1_CITIES:
        return "tier1"
    if city in TIER_2_CITIES:
        return "tier2"
    return "tier3"


def _hobby_tier(hobby):
    for tier, hobbies in HOBBY_TIERS.items():
        if hobby in hobbies:
            return tier
    return "moderate"


def _hobby_group(hobby):
    for group, hobbies in HOBBY_COOCCURRENCE_GROUPS.items():
        if hobby in hobbies:
            return group
    return None


def _select_hobby_cluster(age, education_level, profession, city, income, marital_status):
    """
    Select a lifestyle cluster with soft demographic affinities.

    No cluster is deterministic. Even a young metro engineer can get quiet-home
    hobbies, and an older profile can still travel or use internet-pop hobbies.
    """
    city_tier = _city_tier(city)
    p = (profession or "").lower()
    weights = {
        "quiet_home": 1.20,
        "traditional_family": 0.95,
        "urban_young": 0.75,
        "fitness": 0.70,
        "intellectual": 0.70,
        "creative": 0.65,
        "outdoors": 0.60,
        "regional_cultural": 0.55,
        "finance_status": 0.25,
    }

    if age <= 29:
        weights["urban_young"] *= 2.0
        weights["fitness"] *= 1.35
    elif age >= 38:
        weights["traditional_family"] *= 1.55
        weights["quiet_home"] *= 1.25
        weights["regional_cultural"] *= 1.35

    if city_tier == "tier1":
        weights["urban_young"] *= 1.45
        weights["outdoors"] *= 1.20
        weights["finance_status"] *= 1.30
    elif city_tier in ("tier3", "rural"):
        weights["traditional_family"] *= 1.30
        weights["regional_cultural"] *= 1.25
        weights["urban_young"] *= 0.75

    if education_level in ("Post Graduate", "PhD"):
        weights["intellectual"] *= 1.70
    elif education_level in ("10th", "12th"):
        weights["traditional_family"] *= 1.15
        weights["quiet_home"] *= 1.10

    if any(k in p for k in ["software", "engineer", "data scientist", "product"]):
        weights["urban_young"] *= 1.35
        weights["fitness"] *= 1.10
    if any(k in p for k in ["doctor", "government", "ias", "professor", "research"]):
        weights["intellectual"] *= 1.45
        weights["regional_cultural"] *= 1.15
    if any(k in p for k in ["teacher", "marketing", "hr", "artist"]):
        weights["creative"] *= 1.45
    if any(k in p for k in ["finance", "accountant", "bank", "business"]):
        weights["finance_status"] *= 2.10

    if income >= 15:
        weights["finance_status"] *= 1.65
        weights["outdoors"] *= 1.35
        weights["urban_young"] *= 1.15
    elif income < 4:
        weights["quiet_home"] *= 1.35
        weights["traditional_family"] *= 1.20

    if marital_status in ("divorced", "widowed", "separated"):
        weights["traditional_family"] *= 1.35
        weights["quiet_home"] *= 1.25

    return _weighted_pick(_normalize(weights))


def _compatible_secondary_cluster(primary_cluster):
    compatibility = {
        "urban_young": ["fitness", "creative", "outdoors", "quiet_home"],
        "traditional_family": ["quiet_home", "regional_cultural", "creative"],
        "outdoors": ["fitness", "creative", "urban_young"],
        "intellectual": ["quiet_home", "creative", "finance_status"],
        "creative": ["urban_young", "intellectual", "regional_cultural", "quiet_home"],
        "fitness": ["urban_young", "outdoors", "quiet_home"],
        "regional_cultural": ["traditional_family", "creative", "quiet_home"],
        "finance_status": ["intellectual", "urban_young", "outdoors"],
        "quiet_home": ["traditional_family", "intellectual", "creative"],
    }
    options = compatibility.get(primary_cluster, ["quiet_home"])
    return random.choice(options)


def _cluster_hobby_distribution(clusters):
    dist = {}
    for cluster in clusters:
        for hobby, weight in HOBBY_CLUSTERS[cluster]["hobbies"].items():
            dist[hobby] = dist.get(hobby, 0) + weight

    # Real profiles are often boring. Add a cliche baseline to every cluster so
    # common hobbies remain frequent even for niche lifestyle personas.
    for hobby, weight in COMMON_HOBBIES.items():
        dist[hobby] = dist.get(hobby, 0) + weight * 0.45

    # Add a faint global long-tail floor. Without this, selected clusters can
    # crowd out niche/rare hobbies and make profiles too repetitive.
    for hobby, weight in MODERATELY_COMMON_HOBBIES.items():
        dist[hobby] = dist.get(hobby, 0) + weight * 0.12
    for hobby, weight in NICHE_HOBBIES.items():
        dist[hobby] = dist.get(hobby, 0) + weight * 0.18
    for hobby, weight in RARE_HOBBIES.items():
        dist[hobby] = dist.get(hobby, 0) + weight * 0.22
    return dist


def _apply_demographic_hobby_bias(dist, age, profession, city, income, marital_status):
    p = (profession or "").lower()
    city_tier = _city_tier(city)
    adjusted = dict(dist)

    def boost(hobby, factor):
        if hobby in adjusted:
            adjusted[hobby] *= factor

    if age <= 29:
        for hobby in ["Gaming", "Gym", "Football", "Anime", "K-drama", "Cafe hopping"]:
            boost(hobby, 1.45)
    if age >= 38:
        for hobby in ["Gardening", "Devotional music", "Temple visits", "Spending time with family"]:
            boost(hobby, 1.45)

    if city_tier == "tier1":
        for hobby in ["Cafe hopping", "Photography", "Gym", "Food blogging", "Stock market investing"]:
            boost(hobby, 1.35)
    elif city_tier in ("tier3", "rural"):
        for hobby in ["Cricket watching", "Temple visits", "Cooking", "Spending time with family"]:
            boost(hobby, 1.30)

    if any(k in p for k in ["software", "engineer", "data scientist"]):
        for hobby in ["Gaming", "Gym", "Chess", "Podcast listening"]:
            boost(hobby, 1.25)
    if any(k in p for k in ["doctor", "professor", "research", "scientist"]):
        for hobby in ["Reading", "Research reading", "Yoga", "Meditation"]:
            boost(hobby, 1.35)
    if any(k in p for k in ["finance", "accountant", "bank", "business"]):
        for hobby in ["Stock market investing", "Investing", "Reading"]:
            boost(hobby, 1.45)

    if income >= 15:
        for hobby in ["Travelling", "Photography", "Cafe hopping", "Investing"]:
            boost(hobby, 1.30)
    if marital_status in ("divorced", "widowed", "separated"):
        for hobby in ["Spending time with family", "Listening to music", "Reading"]:
            boost(hobby, 1.20)

    return adjusted


def _apply_cooccurrence_bias(dist, selected_hobbies):
    if not selected_hobbies:
        return dist

    selected_groups = {_hobby_group(h) for h in selected_hobbies}
    selected_groups.discard(None)
    adjusted = {}

    for hobby, weight in dist.items():
        if hobby in selected_hobbies:
            continue

        group = _hobby_group(hobby)
        new_weight = weight
        if group in selected_groups:
            new_weight *= 1.75

        incompatible = set()
        for selected_group in selected_groups:
            incompatible.update(HOBBY_INCOMPATIBLE_GROUPS.get(selected_group, []))
        if group in incompatible:
            new_weight *= 0.28

        adjusted[hobby] = new_weight

    return adjusted


def _sample_hobby_from_tier(dist, tier):
    tier_adjusted = {}
    for hobby, weight in dist.items():
        hobby_tier = _hobby_tier(hobby)
        if hobby_tier == tier:
            tier_adjusted[hobby] = weight * 2.6
        elif hobby_tier == "common":
            tier_adjusted[hobby] = weight * 0.9
        else:
            tier_adjusted[hobby] = weight * 0.35
    return _weighted_pick(_normalize(tier_adjusted))


def _generate_hobbies(age, education_level, profession, city, income, marital_status):
    primary_cluster = _select_hobby_cluster(
        age, education_level, profession, city, income, marital_status
    )
    clusters = [primary_cluster]
    if random.random() < 0.42:
        clusters.append(_compatible_secondary_cluster(primary_cluster))

    base_dist = _cluster_hobby_distribution(clusters)
    base_dist = _apply_demographic_hobby_bias(
        base_dist, age, profession, city, income, marital_status
    )

    n_hobbies = random.choices([3, 4, 5, 6], weights=[0.34, 0.38, 0.20, 0.08], k=1)[0]
    selected = []
    for _ in range(n_hobbies):
        tier = _weighted_pick(HOBBY_TIER_WEIGHTS)
        candidate_dist = _apply_cooccurrence_bias(base_dist, selected)
        if not candidate_dist:
            break
        selected.append(_sample_hobby_from_tier(candidate_dist, tier))

    return selected


def generate_hobbies(age, education_level, profession, city, income, marital_status):
    """
    Public hobby generator returning the existing schema shape: list[str].
    """
    return _generate_hobbies(
        age=age,
        education_level=education_level,
        profession=profession,
        city=city,
        income=income,
        marital_status=marital_status,
    )
