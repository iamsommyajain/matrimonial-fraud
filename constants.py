"""
constants.py
All domain-specific lookup tables for Indian matrimonial profiles.
Keeping these separate means you can expand the vocabulary without
touching any generation logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

import numpy as np

# ── Demographic pools ─────────────────────────────────────────────────────────

RELIGIONS = ["Hindu", "Muslim", "Sikh", "Christian", "Jain", "Buddhist", "Parsi"]
RELIGION_WEIGHTS = [0.79, 0.12, 0.02, 0.02, 0.02, 0.01, 0.01]  # approx India census

RELIGION_STRUCTURE = {
    "Hindu": {
        "sects": [
            "Shaiva",
            "Vaishnava",
            "Shakta",
            "Smarta",
            "Not specified"
        ],

        "communities": {
            "North": [
                "Brahmin",
                "Rajput",
                "Kayastha",
                "Bania",
                "Jat",
                "Yadav",
                "Gujjar",
                "Kurmi",
            ],

            "South": [
                "Iyer",
                "Iyengar",
                "Nair",
                "Ezhava",
                "Reddy",
                "Kamma",
                "Lingayat",
                "Vokkaliga",
            ],

            "West": [
                "Maratha",
                "CKP",
                "Patidar",
                "Lohana",
                "Brahmin",
            ],

            "East": [
                "Kayastha",
                "Mahishya",
                "Baidya",
                "Brahmin",
            ]
        }
    },

    "Muslim": {
        "sects": [
            "Sunni",
            "Shia",
            "Ahmadiyya",
            "Not specified"
        ],

        "communities": [
            "Syed",
            "Ansari",
            "Qureshi",
            "Memon",
            "Pathan",
            "Sheikh",
            "Bohra",
        ]
    },

    "Sikh": {
        "communities": [
            "Jat",
            "Khatri",
            "Arora",
            "Ramgarhia",
        ]
    },

    "Christian": {
        "denominations": [
            "Catholic",
            "Protestant",
            "Orthodox",
            "Pentecostal",
        ],

        "communities": [
            "Syrian Christian",
            "Latin Catholic",
            "Anglo-Indian",
        ]
    },

    "Jain": {
        "sects": [
            "Digambara",
            "Shvetambara",
        ]
    },

    "Buddhist": {
        "sects": [
            "Mahayana",
            "Theravada",
            "Navayana",
        ]
    },

    "Parsi": {
        "communities": [
            "Parsi"
        ]
    }
}

SOCIAL_CATEGORY = [
    "General",
    "OBC",
    "SC",
    "ST",
    "Not specified"
]

MOTHER_TONGUES = ["Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Gujarati",
                  "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", "Assamese"]
TONGUE_WEIGHTS = [0.43, 0.08, 0.08, 0.07, 0.07, 0.05, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02]

GENDERS = ["Male", "Female"]
GENDER_WEIGHTS = [0.55, 0.45]  # slight male skew typical on matrimonial sites

# ── Education ─────────────────────────────────────────────────────────────────

EDUCATION_LEVELS = ["10th", "12th", "Graduate", "Post Graduate", "PhD"]
EDUCATION_WEIGHTS = [0.03, 0.07, 0.50, 0.35, 0.05]

EDUCATION_FIELDS = {
    "10th":          ["General"],
    "12th":          ["Science", "Commerce", "Arts"],
    "Graduate":      ["Engineering", "Medicine", "Commerce", "Arts", "Science",
                      "Law", "Architecture", "Pharmacy", "Nursing"],
    "Post Graduate": ["MBA", "M.Tech", "M.Sc", "MA", "MCA", "LLM", "MD", "MS", "Analytics", "Hospital Administration"],
    "PhD":           ["Engineering", "Sciences", "Humanities", "Medicine", "Management"],
}

# ─────────────────────────────────────────────────────────────
# College hierarchy
# ─────────────────────────────────────────────────────────────

COLLEGES = {

    "Graduate": {

        "elite_engineering": [
            "IIT Delhi",
            "IIT Bombay",
            "IIT Madras",
            "BITS Pilani",
            "NIT Trichy",
            "IIIT Hyderabad",
        ],

        "top_universities": [
            "Delhi University",
            "Jadavpur University",
            "Anna University",
            "Mumbai University",
            "Jamia Millia Islamia",
        ],

        "private_reputed": [
            "Manipal University",
            "VIT Vellore",
            "SRM Institute of Science and Technology",
            "Christ University",
            "Symbiosis International University",
        ],

        "regional_state": [
            "Government Engineering College",
            "State University",
            "Government Degree College",
        ],

        "local_private": [
            "Private Engineering College",
            "Private Degree College",
            "Local Arts and Science College",
        ],
    },

    "Post Graduate": {

        "elite_mba": [
            "IIM Ahmedabad",
            "IIM Bangalore",
            "IIM Calcutta",
            "XLRI Jamshedpur",
        ],

        "technical_pg": [
            "IIT Delhi (M.Tech)",
            "IIT Bombay (M.Tech)",
            "NIT Trichy (M.Tech)",
        ],

        "medical_pg": [
            "AIIMS Delhi",
            "CMC Vellore",
        ],

        "general_pg": [
            "JNU",
            "University of Hyderabad",
            "Delhi University (MA/MSc)",
            "Private University PG",
            "State University PG Department",
        ],
    },

    "PhD": {

        "top_research": [
            "IISc Bangalore",
            "TIFR Mumbai",
            "ISI Kolkata",
            "IIT Delhi",
            "IIT Bombay",
        ],

        "good_universities": [
            "Jawaharlal Nehru University",
            "University of Hyderabad",
            "Anna University",
            "Jadavpur University",
        ],

        "generic": [
            "State University",
            "Private University",
            "Deemed University",
        ],
    },

    "10th": {
        "school": [
            "Kendriya Vidyalaya",
            "DAV Public School",
            "Government Senior Secondary School",
            "Local Government School",
        ]
    },

    "12th": {
        "school": [
            "Kendriya Vidyalaya",
            "DAV Public School",
            "Delhi Public School",
            "Government Junior College",
            "Local Junior College",
        ]
    },
}

COLLEGE_CATEGORY_WEIGHTS = {

    "Graduate": {
        "elite_engineering": 0.03,
        "top_universities": 0.15,
        "private_reputed": 0.18,
        "regional_state": 0.34,
        "local_private": 0.30,
    },

    "Post Graduate": {
        "elite_mba": 0.08,
        "technical_pg": 0.12,
        "medical_pg": 0.07,
        "general_pg": 0.73,
    },

    "PhD": {
        "top_research": 0.18,
        "good_universities": 0.32,
        "generic": 0.50,
    },

    "10th": {
        "school": 1.0,
    },

    "12th": {
        "school": 1.0,
    }
}

COLLEGE_METADATA = {}

class MigrationProfile(TypedDict):
    metro_migration_affinity: float
    international_migration_affinity: float
    career_mobility: float
    marriage_relocation_openness: float

class CollegeMetadata(TypedDict, total=False):
    tier: str
    fields: List[str]
    field_distribution: Dict[str, float]
    city: Optional[str]
    category: str
    admission_selectivity: float
    platform_representation: float
    prestige_score: float
    avg_salary_multiplier: float
    migration_profile: MigrationProfile
    acceptance_weight: float
    migration_affinity: float

COLLEGE_FIELD_DISTRIBUTIONS = {
    "elite_engineering": {
        "Engineering": 0.55,
        "Computer Science": 0.35,
        "Science": 0.10,
    },
    "top_universities": {
        "Arts": 0.32,
        "Science": 0.28,
        "Commerce": 0.18,
        "Law": 0.12,
        "Engineering": 0.06,
        "Management": 0.04,
    },
    "private_reputed": {
        "Engineering": 0.38,
        "Management": 0.24,
        "Commerce": 0.16,
        "Science": 0.12,
        "Arts": 0.08,
        "Law": 0.02,
    },
    "regional_state": {
        "Engineering": 0.26,
        "Science": 0.22,
        "Commerce": 0.18,
        "Arts": 0.18,
        "Law": 0.08,
        "Management": 0.08,
    },
    "local_private": {
        "Engineering": 0.20,
        "Science": 0.18,
        "Commerce": 0.22,
        "Arts": 0.24,
        "Law": 0.06,
        "Management": 0.10,
    },
    "elite_mba": {
        "MBA": 0.72,
        "Management": 0.18,
        "Commerce": 0.10,
    },
    "technical_pg": {
        "M.Tech": 0.60,
        "M.Sc": 0.20,
        "MBA": 0.10,
        "Analytics": 0.10,
    },
    "medical_pg": {
        "MD": 0.46,
        "MS": 0.34,
        "M.Sc": 0.12,
        "Hospital Administration": 0.08,
    },
    "general_pg": {
        "MA": 0.28,
        "M.Sc": 0.24,
        "MBA": 0.18,
        "MCA": 0.12,
        "LLM": 0.08,
        "Analytics": 0.10,
    },
    "top_research": {
        "Sciences": 0.38,
        "Engineering": 0.30,
        "Humanities": 0.18,
        "Management": 0.14,
    },
    "good_universities": {
        "Arts": 0.30,
        "Science": 0.30,
        "Commerce": 0.18,
        "Law": 0.10,
        "Management": 0.12,
    },
    "generic": {
        "Arts": 0.26,
        "Science": 0.24,
        "Commerce": 0.18,
        "Law": 0.12,
        "Management": 0.10,
        "MCA": 0.10,
    },
    "school": {
        "General": 0.5,
        "Science": 0.25,
        "Commerce": 0.15,
        "Arts": 0.10,
    },
}

COLLEGE_OVERRIDES = {
    "IIT Delhi": {
        "tier": "elite",
        "city": "Delhi",
        "category": "elite_engineering",
        "admission_selectivity": 0.010,
        "platform_representation": 0.055,
        "prestige_score": 0.95,
        "avg_salary_multiplier": 2.9,
        "field_distribution": {
            "Engineering": 0.58,
            "Computer Science": 0.42,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.82,
            "international_migration_affinity": 0.22,
            "career_mobility": 0.78,
            "marriage_relocation_openness": 0.60,
        },
    },
    "IIT Bombay": {
        "tier": "elite",
        "city": "Mumbai",
        "category": "elite_engineering",
        "admission_selectivity": 0.010,
        "platform_representation": 0.050,
        "prestige_score": 0.95,
        "avg_salary_multiplier": 2.9,
        "field_distribution": {
            "Engineering": 0.55,
            "Computer Science": 0.45,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.80,
            "international_migration_affinity": 0.24,
            "career_mobility": 0.80,
            "marriage_relocation_openness": 0.58,
        },
    },
    "IIT Madras": {
        "tier": "elite",
        "city": "Chennai",
        "category": "elite_engineering",
        "admission_selectivity": 0.010,
        "platform_representation": 0.045,
        "prestige_score": 0.94,
        "avg_salary_multiplier": 2.85,
        "field_distribution": {
            "Engineering": 0.60,
            "Computer Science": 0.40,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.78,
            "international_migration_affinity": 0.20,
            "career_mobility": 0.75,
            "marriage_relocation_openness": 0.55,
        },
    },
    "IIIT Hyderabad": {
        "tier": "elite",
        "city": "Hyderabad",
        "category": "elite_engineering",
        "admission_selectivity": 0.015,
        "platform_representation": 0.055,
        "prestige_score": 0.92,
        "avg_salary_multiplier": 2.8,
        "field_distribution": {
            "Engineering": 0.45,
            "Computer Science": 0.40,
            "Science": 0.15,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.84,
            "international_migration_affinity": 0.18,
            "career_mobility": 0.76,
            "marriage_relocation_openness": 0.62,
        },
    },
    "Ashoka University": {
        "tier": "premium",
        "city": "Sonepat",
        "category": "top_universities",
        "admission_selectivity": 0.06,
        "platform_representation": 0.025,
        "prestige_score": 0.88,
        "avg_salary_multiplier": 1.7,
        "field_distribution": {
            "Arts": 0.35,
            "Science": 0.25,
            "Commerce": 0.20,
            "Management": 0.10,
            "Law": 0.10,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.65,
            "international_migration_affinity": 0.28,
            "career_mobility": 0.58,
            "marriage_relocation_openness": 0.60,
        },
    },
    "ISI Kolkata": {
        "tier": "premium",
        "city": "Kolkata",
        "category": "top_research",
        "admission_selectivity": 0.05,
        "platform_representation": 0.020,
        "prestige_score": 0.90,
        "avg_salary_multiplier": 2.1,
        "field_distribution": {
            "Sciences": 0.64,
            "Engineering": 0.18,
            "Humanities": 0.12,
            "Management": 0.06,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.70,
            "international_migration_affinity": 0.32,
            "career_mobility": 0.62,
            "marriage_relocation_openness": 0.50,
        },
    },
    "NLSIU": {
        "tier": "elite",
        "city": "Bangalore",
        "category": "top_universities",
        "admission_selectivity": 0.018,
        "platform_representation": 0.022,
        "prestige_score": 0.91,
        "avg_salary_multiplier": 2.5,
        "field_distribution": {
            "Law": 0.82,
            "Commerce": 0.10,
            "Management": 0.08,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.78,
            "international_migration_affinity": 0.22,
            "career_mobility": 0.71,
            "marriage_relocation_openness": 0.53,
        },
    },
    "SPJIMR": {
        "tier": "premium",
        "city": "Mumbai",
        "category": "elite_mba",
        "admission_selectivity": 0.07,
        "platform_representation": 0.035,
        "prestige_score": 0.89,
        "avg_salary_multiplier": 2.15,
        "field_distribution": {
            "MBA": 0.72,
            "Management": 0.18,
            "Commerce": 0.10,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.70,
            "international_migration_affinity": 0.24,
            "career_mobility": 0.65,
            "marriage_relocation_openness": 0.57,
        },
    },
    "FMS Delhi": {
        "tier": "premium",
        "city": "Delhi",
        "category": "elite_mba",
        "admission_selectivity": 0.06,
        "platform_representation": 0.030,
        "prestige_score": 0.88,
        "avg_salary_multiplier": 2.1,
        "field_distribution": {
            "MBA": 0.65,
            "Management": 0.22,
            "Commerce": 0.13,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.71,
            "international_migration_affinity": 0.27,
            "career_mobility": 0.64,
            "marriage_relocation_openness": 0.58,
        },
    },
    "BITS Pilani Dubai": {
        "tier": "premium",
        "city": "Dubai",
        "category": "elite_engineering",
        "admission_selectivity": 0.028,
        "platform_representation": 0.015,
        "prestige_score": 0.84,
        "avg_salary_multiplier": 2.0,
        "field_distribution": {
            "Engineering": 0.50,
            "Computer Science": 0.40,
            "Science": 0.10,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.72,
            "international_migration_affinity": 0.42,
            "career_mobility": 0.68,
            "marriage_relocation_openness": 0.66,
        },
    },
    "Jadavpur University": {
        "tier": "premium",
        "city": "Kolkata",
        "category": "top_universities",
        "admission_selectivity": 0.085,
        "platform_representation": 0.035,
        "prestige_score": 0.76,
        "avg_salary_multiplier": 1.5,
        "field_distribution": {
            "Engineering": 0.32,
            "Science": 0.24,
            "Arts": 0.20,
            "Commerce": 0.14,
            "Law": 0.05,
            "Management": 0.05,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.62,
            "international_migration_affinity": 0.16,
            "career_mobility": 0.58,
            "marriage_relocation_openness": 0.53,
        },
    },
    "VIT Vellore": {
        "tier": "upper_mid",
        "city": "Vellore",
        "category": "private_reputed",
        "admission_selectivity": 0.35,
        "platform_representation": 0.048,
        "prestige_score": 0.58,
        "avg_salary_multiplier": 1.35,
        "field_distribution": {
            "Engineering": 0.46,
            "Science": 0.18,
            "Commerce": 0.14,
            "Arts": 0.10,
            "Management": 0.12,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.48,
            "international_migration_affinity": 0.12,
            "career_mobility": 0.47,
            "marriage_relocation_openness": 0.45,
        },
    },
    "SRM Institute of Science and Technology": {
        "tier": "upper_mid",
        "city": "Chennai",
        "category": "private_reputed",
        "admission_selectivity": 0.38,
        "platform_representation": 0.045,
        "prestige_score": 0.54,
        "avg_salary_multiplier": 1.3,
        "field_distribution": {
            "Engineering": 0.44,
            "Science": 0.16,
            "Commerce": 0.14,
            "Arts": 0.12,
            "Management": 0.14,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.50,
            "international_migration_affinity": 0.10,
            "career_mobility": 0.49,
            "marriage_relocation_openness": 0.47,
        },
    },
    "Manipal University": {
        "tier": "upper_mid",
        "city": "Manipal",
        "category": "private_reputed",
        "admission_selectivity": 0.42,
        "platform_representation": 0.052,
        "prestige_score": 0.56,
        "avg_salary_multiplier": 1.34,
        "field_distribution": {
            "Engineering": 0.32,
            "Science": 0.20,
            "Commerce": 0.18,
            "Arts": 0.10,
            "Management": 0.20,
        },
        "migration_profile": {
            "metro_migration_affinity": 0.45,
            "international_migration_affinity": 0.14,
            "career_mobility": 0.44,
            "marriage_relocation_openness": 0.46,
        },
    },
}

COLLEGE_TIER_META = {
    "elite": {
        "admission_selectivity": 0.01,
        "platform_representation": 0.04,
        "prestige_score": 0.94,
        "avg_salary_multiplier": 2.8,
        "migration_profile": {
            "metro_migration_affinity": 0.78,
            "international_migration_affinity": 0.30,
            "career_mobility": 0.75,
            "marriage_relocation_openness": 0.63,
        },
    },
    "premium": {
        "admission_selectivity": 0.05,
        "platform_representation": 0.12,
        "prestige_score": 0.76,
        "avg_salary_multiplier": 1.9,
        "migration_profile": {
            "metro_migration_affinity": 0.65,
            "international_migration_affinity": 0.19,
            "career_mobility": 0.60,
            "marriage_relocation_openness": 0.52,
        },
    },
    "upper_mid": {
        "admission_selectivity": 0.14,
        "platform_representation": 0.22,
        "prestige_score": 0.57,
        "avg_salary_multiplier": 1.45,
        "migration_profile": {
            "metro_migration_affinity": 0.50,
            "international_migration_affinity": 0.12,
            "career_mobility": 0.44,
            "marriage_relocation_openness": 0.42,
        },
    },
    "mid": {
        "admission_selectivity": 0.30,
        "platform_representation": 0.28,
        "prestige_score": 0.36,
        "avg_salary_multiplier": 1.08,
        "migration_profile": {
            "metro_migration_affinity": 0.34,
            "international_migration_affinity": 0.07,
            "career_mobility": 0.28,
            "marriage_relocation_openness": 0.30,
        },
    },
    "local": {
        "admission_selectivity": 0.55,
        "platform_representation": 0.34,
        "prestige_score": 0.18,
        "avg_salary_multiplier": 0.80,
        "migration_profile": {
            "metro_migration_affinity": 0.22,
            "international_migration_affinity": 0.03,
            "career_mobility": 0.14,
            "marriage_relocation_openness": 0.20,
        },
    },
}

COHORT_METADATA = {
    "young_professional": {
        "age_range": (22, 27),
        "startup_bias": 1.40,
        "legacy_it_bias": 0.80,
        "government_bias": 0.75,
        "remote_work_bias": 0.88,
        "migration_bias": 1.05,
    },
    "mid_career": {
        "age_range": (28, 35),
        "startup_bias": 1.15,
        "legacy_it_bias": 1.00,
        "government_bias": 0.90,
        "remote_work_bias": 0.76,
        "migration_bias": 0.95,
    },
    "established_leadership": {
        "age_range": (36, 50),
        "startup_bias": 0.75,
        "legacy_it_bias": 1.20,
        "government_bias": 1.10,
        "remote_work_bias": 0.60,
        "migration_bias": 0.82,
    },
}

EMAIL_DOMAIN_COHORT_PREFERENCES = {
    "young": {"personal": 0.80, "institutional": 0.10, "privacy_focused": 0.07, "legacy": 0.03},
    "mid":   {"personal": 0.76, "institutional": 0.13, "privacy_focused": 0.07, "legacy": 0.04},
    "older": {"personal": 0.70, "institutional": 0.18, "privacy_focused": 0.06, "legacy": 0.06},
}

EMAIL_INSTITUTIONAL_BOOST = {
    "doctor": 0.18,
    "professor": 0.16,
    "scientist": 0.14,
    "government": 0.13,
    "ias": 0.12,
    "ips": 0.12,
    "lawyer": 0.08,
}

EMAIL_DOMAIN_LEGACY_PROVIDERS = [
    "yahoo.com",
    "hotmail.com",
    "rediffmail.com",
    "live.com",
]

COLLEGE_TIER_CATEGORY_MAP = {
    "elite_engineering",
    "elite_mba",
    "top_universities",
    "top_research",
    "private_reputed",
    "technical_pg",
    "medical_pg",
    "general_pg",
    "good_universities",
    "regional_state",
    "generic",
    "local_private",
    "school",
}

def _normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values())
    if total <= 0:
        return {key: 1.0 / len(dist) for key in dist}
    return {key: value / total for key, value in dist.items()}

def _category_tier(category: str) -> str:
    if category in {"elite_engineering", "elite_mba", "top_research"}:
        return "elite"
    if category in {"top_universities", "good_universities", "technical_pg", "medical_pg"}:
        return "premium"
    if category in {"private_reputed", "general_pg"}:
        return "upper_mid"
    if category in {"regional_state", "generic"}:
        return "mid"
    return "local"

for level, categories in COLLEGES.items():
    for category, names in categories.items():
        for name in names:
            override = COLLEGE_OVERRIDES.get(name, {})
            tier = override.get("tier") or _category_tier(category)
            tier_meta = COLLEGE_TIER_META[tier]
            field_distribution = override.get(
                "field_distribution",
                COLLEGE_FIELD_DISTRIBUTIONS.get(category, {field: 1.0 for field in EDUCATION_FIELDS[level]})
            )
            field_distribution = _normalize_distribution({
                f: v for f, v in field_distribution.items() if f in EDUCATION_FIELDS[level]
            })
            if not field_distribution:
                field_distribution = _normalize_distribution({field: 1.0 for field in EDUCATION_FIELDS[level]})
            fields = list(field_distribution)
            admission_selectivity = override.get("admission_selectivity", tier_meta["admission_selectivity"])
            platform_representation = override.get("platform_representation", tier_meta["platform_representation"])
            prestige_score = override.get("prestige_score", tier_meta["prestige_score"])
            avg_salary_multiplier = override.get("avg_salary_multiplier", tier_meta["avg_salary_multiplier"])
            migration_profile = override.get("migration_profile", tier_meta["migration_profile"])
            migration_affinity = (
                0.4 * migration_profile["metro_migration_affinity"]
                + 0.35 * migration_profile["career_mobility"]
                + 0.25 * migration_profile["international_migration_affinity"]
            )
            COLLEGE_METADATA[name] = {
                "tier": tier,
                "fields": fields,
                "field_distribution": field_distribution,
                "city": override.get("city"),
                "category": category,
                "admission_selectivity": admission_selectivity,
                "platform_representation": platform_representation,
                "prestige_score": prestige_score,
                "avg_salary_multiplier": avg_salary_multiplier,
                "migration_profile": migration_profile,
                "migration_affinity": migration_affinity,
                "acceptance_weight": admission_selectivity,
            }

COLLEGE_METADATA.update({
    name: {**COLLEGE_METADATA.get(name, {}), **override}
    for name, override in COLLEGE_OVERRIDES.items()
})

def validate_metadata_consistency() -> List[str]:
    errors = []

    for college_name, metadata in COLLEGE_METADATA.items():
        if metadata.get("tier") not in COLLEGE_TIER_META:
            errors.append(f"Invalid tier for college {college_name}: {metadata.get('tier')}")
        if not (0 <= metadata.get("admission_selectivity", 0) <= 1):
            errors.append(f"Invalid admission_selectivity for {college_name}")
        if not (0 <= metadata.get("platform_representation", 0) <= 1):
            errors.append(f"Invalid platform_representation for {college_name}")
        if not (0 <= metadata.get("prestige_score", 0) <= 1):
            errors.append(f"Invalid prestige_score for {college_name}")
        field_dist = metadata.get("field_distribution", {})
        if abs(sum(field_dist.values()) - 1.0) > 1e-6:
            errors.append(f"Field distribution does not sum to 1 for {college_name}")

    for company_name, metadata in COMPANY_METADATA.items():
        if metadata.get("tier") is None:
            errors.append(f"Company metadata missing tier for {company_name}")
        if metadata.get("industry") is None:
            errors.append(f"Company metadata missing industry for {company_name}")
        if not (0 <= metadata.get("remote_work_affinity", 0) <= 1):
            errors.append(f"Invalid remote_work_affinity for {company_name}")
        if not (0 <= metadata.get("prestige_score", 0) <= 1):
            errors.append(f"Invalid prestige_score for {company_name}")
        if not (0 <= metadata.get("work_life_balance", 0) <= 1):
            errors.append(f"Invalid work_life_balance for {company_name}")
        sd = metadata.get("salary_distribution", {})
        if sd and sd.get("sigma_adjustment", 0) < 0:
            errors.append(f"Invalid salary_distribution sigma_adjustment for {company_name}")

    for cohort_name, cohort_meta in COHORT_METADATA.items():
        if cohort_name not in {"young_professional", "mid_career", "established_leadership"}:
            errors.append(f"Unexpected cohort label: {cohort_name}")

    for domain_type, domains in EMAIL_DOMAIN_CATEGORIES.items():
        if not domains:
            errors.append(f"Empty email domain category: {domain_type}")

    return errors

# ── Profession ────────────────────────────────────────────────────────────────

# Maps education level → realistic profession pool
PROFESSIONS_BY_EDUCATION = {
    "10th":          ["Shop Owner", "Daily Wage Worker", "Driver", "Security Guard", "Domestic Worker"],
    "12th":          ["Sales Executive", "Office Assistant", "Data Entry Operator",
                      "Shop Owner", "Delivery Executive", "Bank Clerk"],
    "Graduate":      ["Software Engineer", "Teacher", "Bank Officer", "HR Executive",
                      "Marketing Executive", "Accountant", "Civil Engineer", "Pharmacist",
                      "Nurse", "Government Employee", "Business Owner", "Sales Manager"],
    "Post Graduate": ["Senior Software Engineer", "Product Manager", "Business Analyst",
                      "Doctor", "Lawyer", "Professor", "Finance Manager", "Consultant",
                      "Research Scientist", "Government IAS/IPS", "Chartered Accountant"],
    "PhD":           ["Professor", "Research Scientist", "Data Scientist", "Consultant",
                      "Senior Research Fellow", "Principal Engineer"],
}

# Realistic company names by profession category
COMPANIES_LEGITIMATE = {
    "tech":     ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Accenture India",
                 "IBM India", "Capgemini", "Cognizant", "L&T Infotech", "Mphasis",
                 "Hexaware", "Mindtree", "Persistent Systems", "NIIT Technologies", "Goldman Sachs", "Amazon India", "Google India", "Microsoft India", "Meta",
                 "Apple India", "Flipkart", "Zomato", "Swiggy", "Paytm", "Ola", "Uber India", "LinkedIn", "Adobe India", "Salesforce India", "Oracle India", "SAP India",
                 "Samsung", "Razorpay", "Zerodha", "CRED", "Groww", "Meesho", "Zepto", "PhonePe"],
    "finance":  ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra",
                 "HDFC Life", "LIC", "Deloitte", "PwC India", "KPMG India", "EY India"],
    "govt":     ["Central Government", "State Government", "PSU Company", "DRDO",
                 "ISRO", "Indian Railways", "Indian Army", "Indian Police Service"],
    "medical":  ["Apollo Hospitals", "Fortis Healthcare", "Max Healthcare", "AIIMS",
                 "Private Clinic", "Government Hospital", "Medanta"],
    "education":["IIT", "NIT", "Private University", "Government College", "CBSE School"],
    "business": ["Family Business", "Self Employed", "Partnership Firm", "Proprietorship"],
}

# Email domains are categorized for soft risk modeling. Fraud should not be a
# simple domain blacklist: many fraud profiles still use normal personal mail,
# and legitimate users may use privacy-focused or older providers.
EMAIL_DOMAIN_CATEGORIES = {
    "personal": [
        "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
        "rediffmail.com", "icloud.com", "live.com",
    ],
    "institutional": [
        "infosys.com", "tcs.com", "wipro.com", "hcltech.com",
        "techmahindra.com", "accenture.com", "cognizant.com",
        "hdfcbank.com", "icicibank.com", "sbi.co.in", "axisbank.com",
        "aiims.edu", "apollohospitals.com", "fortishealthcare.com",
        "iitd.ac.in", "iitb.ac.in", "iitm.ac.in", "iisc.ac.in",
        "du.ac.in", "gov.in", "nic.in", "isro.gov.in","igdtuw.ac.in", "jnu.ac.in", "iimb.ac.in", "iimahd.ernet.in",
        "thapar.edu", "manipal.edu", "vit.ac.in", "srmuniv.edu.in", "christuniversity.in", "symbiosis.edu"
    ],
    "privacy_focused": [
        "proton.me", "protonmail.com", "tutanota.com", "zoho.com",
    ],
    "disposable": [
        "mailinator.com", "tempmail.com", "guerrillamail.com", "throwam.com",
        "fakeinbox.com", "yopmail.com", "trashmail.com", "dispostable.com",
    ],
    "typo_squatted": [
        "gmai1.com", "gmial.com", "outllook.com", "yah00.com",
        "hotmai1.com", "iitd-ac.in", "infosys-careers.com",
        "google-hr.co", "aiims-career.org", "tcs-jobs.in",
    ],
    "fake_corporate": [
        "consultantmail.com", "companyhr.co", "career-institute.org",
        "corp-mail.in", "recruiterhub.co", "official-team.com",
    ],
}

# Backward-compatible aliases for older modules/imports. These are no longer
# used as hard fraud labels.
LEGITIMATE_EMAIL_DOMAINS = EMAIL_DOMAIN_CATEGORIES["personal"]
SUSPICIOUS_EMAIL_DOMAINS = (
    EMAIL_DOMAIN_CATEGORIES["disposable"]
    + EMAIL_DOMAIN_CATEGORIES["typo_squatted"]
    + EMAIL_DOMAIN_CATEGORIES["fake_corporate"]
)

COMPANY_DOMAIN_MAP = {
    "Infosys": ["infosys.com"],
    "TCS": ["tcs.com"],
    "Wipro": ["wipro.com"],
    "HCL": ["hcltech.com"],
    "Tech Mahindra": ["techmahindra.com"],
    "Accenture India": ["accenture.com"],
    "Cognizant": ["cognizant.com"],
    "Google India": ["google.com"],
    "Microsoft India": ["microsoft.com"],
    "Meta": ["meta.com"],
    "Apple India": ["apple.com"],
    "Goldman Sachs": ["goldmansachs.com"],
    "Amazon India": ["amazon.in", "amazon.com"],
    "HDFC Bank": ["hdfcbank.com"],
    "ICICI Bank": ["icicibank.com"],
    "SBI": ["sbi.co.in"],
    "Axis Bank": ["axisbank.com"],
    "Kotak Mahindra": ["kotak.com"],
    "AIIMS": ["aiims.edu"],
    "Apollo Hospitals": ["apollohospitals.com"],
    "Fortis Healthcare": ["fortishealthcare.com"],
    "Max Healthcare": ["maxhealthcare.in"],
    "IIT Delhi": ["iitd.ac.in"],
    "IIT Bombay": ["iitb.ac.in"],
    "IIT Madras": ["iitm.ac.in"],
    "IISc Bangalore": ["iisc.ac.in"],
    "Delhi University": ["du.ac.in"],
    "Goldman Sachs": ["goldmansachs.com"],
    "Amazon India": ["amazon.in", "amazon.com"],
    "LinkedIn": ["linkedin.com"],
    "Flipkart": ["flipkart.com"],
    "Ola": ["ola.com"],
    "Uber India": ["uber.com"],
    "Zomato": ["zomato.com"],
    "Swiggy": ["swiggy.com"],
    "Paytm": ["paytm.com"],
    "Adobe India": ["adobe.com"],
    "Salesforce India": ["salesforce.com"],
    "Oracle India": ["oracle.com"],
    "SAP India": ["sap.com"],
    "Samsung": ["samsung.com"],
    "Razorpay": ["razorpay.com"],
    "Zerodha": ["zerodha.com"],
    "CRED": ["cred.club"],
    "Groww": ["groww.in"],
    "Meesho": ["meesho.com"],
    "Zepto": ["zepto.com"],
    "PhonePe": ["phonepe.com"],
    "Government of India": ["gov.in", "nic.in"],
    "Central Government": ["gov.in", "nic.in"],
    "State Government": ["gov.in"],
    "ISRO": ["isro.gov.in"],
    "DRDO": ["drdo.gov.in"],
}

COMPANY_METADATA = {
    "Google India": {
        "tier": "elite",
        "industry": "tech",
        "salary_multiplier": 3.4,
        "city_affinity": ["tier1"],
    },
    "Microsoft India": {
        "tier": "elite",
        "industry": "tech",
        "salary_multiplier": 3.3,
        "city_affinity": ["tier1"],
    },
    "Meta": {
        "tier": "elite",
        "industry": "tech",
        "salary_multiplier": 3.3,
        "city_affinity": ["tier1"],
    },
    "Amazon India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.6,
        "city_affinity": ["tier1"],
    },
    "Salesforce India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.4,
        "city_affinity": ["tier1"],
    },
    "Apple India": {
        "tier": "elite",
        "industry": "tech",
        "salary_multiplier": 3.0,
        "city_affinity": ["tier1"],
    },
    "LinkedIn": {
        "tier": "elite",
        "industry": "tech",
        "salary_multiplier": 2.8,
        "city_affinity": ["tier1"],
    },
    "Amazon India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.6,
        "city_affinity": ["tier1"],
    },
    "Flipkart": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.2,
        "city_affinity": ["tier1", "tier2"],
    },
    "Ola": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.9,
        "city_affinity": ["tier1", "tier2"],
    },
    "Uber India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.1,
        "city_affinity": ["tier1", "tier2"],
    },
    "Zomato": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.4,
        "city_affinity": ["tier1", "tier2"],
    },
    "Swiggy": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.3,
        "city_affinity": ["tier1", "tier2"],
    },
    "Paytm": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.8,
        "city_affinity": ["tier1", "tier2"],
    },
    "SAP India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.2,
        "city_affinity": ["tier1"],
    },
    "Samsung": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.0,
        "city_affinity": ["tier1", "tier2"],
    },
    "Razorpay": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.9,
        "city_affinity": ["tier1"],
    },
    "Zerodha": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.8,
        "city_affinity": ["tier1"],
    },
    "CRED": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.9,
        "city_affinity": ["tier1"],
    },
    "Groww": {
        "tier": "mid",
        "industry": "finance",
        "salary_multiplier": 1.4,
        "city_affinity": ["tier1", "tier2"],
    },
    "Meesho": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.35,
        "city_affinity": ["tier1", "tier2"],
    },
    "Zepto": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.25,
        "city_affinity": ["tier1", "tier2"],
    },
    "PhonePe": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.8,
        "city_affinity": ["tier1", "tier2"],
    },
    "Adobe India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.3,
        "city_affinity": ["tier1"],
    },
    "Oracle India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 2.2,
        "city_affinity": ["tier1"],
    },
    "TCS": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.4,
        "city_affinity": ["tier1", "tier2"],
    },
    "Infosys": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.3,
        "city_affinity": ["tier1", "tier2"],
    },
    "Wipro": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.25,
        "city_affinity": ["tier1", "tier2"],
    },
    "HCL": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1", "tier2"],
    },
    "Tech Mahindra": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1", "tier2"],
    },
    "Accenture India": {
        "tier": "upper_mid",
        "industry": "consulting",
        "salary_multiplier": 1.6,
        "city_affinity": ["tier1", "tier2"],
    },
    "Cognizant": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.4,
        "city_affinity": ["tier1", "tier2"],
    },
    "Capgemini": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.3,
        "city_affinity": ["tier1", "tier2"],
    },
    "IBM India": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.5,
        "city_affinity": ["tier1", "tier2"],
    },
    "L&T Infotech": {
        "tier": "upper_mid",
        "industry": "tech",
        "salary_multiplier": 1.3,
        "city_affinity": ["tier1", "tier2"],
    },
    "Mphasis": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.15,
        "city_affinity": ["tier1", "tier2"],
    },
    "Hexaware": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.1,
        "city_affinity": ["tier1", "tier2"],
    },
    "Mindtree": {
        "tier": "mid",
        "industry": "tech",
        "salary_multiplier": 1.1,
        "city_affinity": ["tier1", "tier2"],
    },
    "HDFC Bank": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.5,
        "city_affinity": ["tier1", "tier2"],
    },
    "Goldman Sachs": {
        "tier": "elite",
        "industry": "finance",
        "salary_multiplier": 2.6,
        "city_affinity": ["tier1"],
    },
    "ICICI Bank": {
        "tier": "upper_mid",
        "industry": "finance",
        "salary_multiplier": 1.5,
        "city_affinity": ["tier1", "tier2"],
    },
    "SBI": {
        "tier": "mid",
        "industry": "finance",
        "salary_multiplier": 1.1,
        "city_affinity": ["tier1", "tier2"],
    },
    "Axis Bank": {
        "tier": "mid",
        "industry": "finance",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1", "tier2"],
    },
    "Kotak Mahindra": {
        "tier": "mid",
        "industry": "finance",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1", "tier2"],
    },
    "LIC": {
        "tier": "mid",
        "industry": "finance",
        "salary_multiplier": 1.05,
        "city_affinity": ["tier1", "tier2"],
    },
    "Deloitte": {
        "tier": "upper_mid",
        "industry": "consulting",
        "salary_multiplier": 1.7,
        "city_affinity": ["tier1"],
    },
    "PwC India": {
        "tier": "upper_mid",
        "industry": "consulting",
        "salary_multiplier": 1.7,
        "city_affinity": ["tier1"],
    },
    "KPMG India": {
        "tier": "upper_mid",
        "industry": "consulting",
        "salary_multiplier": 1.6,
        "city_affinity": ["tier1"],
    },
    "EY India": {
        "tier": "upper_mid",
        "industry": "consulting",
        "salary_multiplier": 1.6,
        "city_affinity": ["tier1"],
    },
    "Central Government": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.0,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "State Government": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 0.95,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "PSU Company": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.05,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "DRDO": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1"],
    },
    "ISRO": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.25,
        "city_affinity": ["tier1"],
    },
    "Indian Railways": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.0,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Indian Army": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 0.95,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Indian Police Service": {
        "tier": "mid",
        "industry": "government",
        "salary_multiplier": 1.1,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Apollo Hospitals": {
        "tier": "upper_mid",
        "industry": "medical",
        "salary_multiplier": 1.8,
        "city_affinity": ["tier1", "tier2"],
    },
    "Fortis Healthcare": {
        "tier": "upper_mid",
        "industry": "medical",
        "salary_multiplier": 1.7,
        "city_affinity": ["tier1", "tier2"],
    },
    "Max Healthcare": {
        "tier": "upper_mid",
        "industry": "medical",
        "salary_multiplier": 1.6,
        "city_affinity": ["tier1", "tier2"],
    },
    "AIIMS": {
        "tier": "elite",
        "industry": "medical",
        "salary_multiplier": 2.6,
        "city_affinity": ["tier1"],
    },
    "Private Clinic": {
        "tier": "local",
        "industry": "medical",
        "salary_multiplier": 1.0,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Government Hospital": {
        "tier": "mid",
        "industry": "medical",
        "salary_multiplier": 0.95,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Medanta": {
        "tier": "upper_mid",
        "industry": "medical",
        "salary_multiplier": 1.55,
        "city_affinity": ["tier1"],
    },
    "Family Business": {
        "tier": "self_employed",
        "industry": "business",
        "salary_multiplier": 1.3,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Self Employed": {
        "tier": "self_employed",
        "industry": "business",
        "salary_multiplier": 1.2,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Partnership Firm": {
        "tier": "self_employed",
        "industry": "business",
        "salary_multiplier": 1.15,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
    "Proprietorship": {
        "tier": "self_employed",
        "industry": "business",
        "salary_multiplier": 1.1,
        "city_affinity": ["tier1", "tier2", "tier3"],
    },
}

for company_name, metadata in COMPANY_METADATA.items():
    metadata.setdefault("remote_work_affinity", 0.35)
    metadata.setdefault("work_life_balance", 0.45)
    metadata.setdefault(
        "prestige_score",
        0.75 if metadata.get("tier") in {"elite", "premium"} else 0.55 if metadata.get("tier") in {"upper_mid", "upper_mid"} else 0.35 if metadata.get("tier") == "mid" else 0.25
    )
    metadata.setdefault("salary_distribution", {"sigma_adjustment": 0.15, "tail_probability": 0.08})


# ── Canonical city → state mapping ──────────────────────────────

CITY_STATE_MAP = {

    # ───────────────── Maharashtra ─────────────────
    "Mumbai": "Maharashtra",
    "Pune": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra",
    "Aurangabad": "Maharashtra",
    "Thane": "Maharashtra",
    "Kolhapur": "Maharashtra",
    "Solapur": "Maharashtra",
    "Amravati": "Maharashtra",
    "Jalgaon": "Maharashtra",

    # ───────────────── Delhi ─────────────────
    "Delhi": "Delhi",
    "New Delhi": "Delhi",

    # ───────────────── Karnataka ─────────────────
    "Bangalore": "Karnataka",
    "Mysore": "Karnataka",
    "Mangalore": "Karnataka",
    "Hubli": "Karnataka",
    "Belgaum": "Karnataka",

    # ───────────────── Tamil Nadu ─────────────────
    "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu",
    "Madurai": "Tamil Nadu",
    "Salem": "Tamil Nadu",
    "Tiruchirappalli": "Tamil Nadu",
    "Vellore": "Tamil Nadu",
    "Erode": "Tamil Nadu",

    # ───────────────── Telangana ─────────────────
    "Hyderabad": "Telangana",
    "Warangal": "Telangana",
    "Nizamabad": "Telangana",
    "Karimnagar": "Telangana",

    # ───────────────── Andhra Pradesh ─────────────────
    "Visakhapatnam": "Andhra Pradesh",
    "Vijayawada": "Andhra Pradesh",
    "Guntur": "Andhra Pradesh",
    "Nellore": "Andhra Pradesh",
    "Kurnool": "Andhra Pradesh",
    "Rajahmundry": "Andhra Pradesh",

    # ───────────────── Gujarat ─────────────────
    "Ahmedabad": "Gujarat",
    "Surat": "Gujarat",
    "Vadodara": "Gujarat",
    "Rajkot": "Gujarat",
    "Bhavnagar": "Gujarat",
    "Jamnagar": "Gujarat",

    # ───────────────── Rajasthan ─────────────────
    "Jaipur": "Rajasthan",
    "Jodhpur": "Rajasthan",
    "Udaipur": "Rajasthan",
    "Kota": "Rajasthan",
    "Ajmer": "Rajasthan",
    "Bikaner": "Rajasthan",

    # ───────────────── Uttar Pradesh ─────────────────
    "Lucknow": "Uttar Pradesh",
    "Kanpur": "Uttar Pradesh",
    "Noida": "Uttar Pradesh",
    "Ghaziabad": "Uttar Pradesh",
    "Agra": "Uttar Pradesh",
    "Meerut": "Uttar Pradesh",
    "Varanasi": "Uttar Pradesh",
    "Prayagraj": "Uttar Pradesh",
    "Aligarh": "Uttar Pradesh",
    "Bareilly": "Uttar Pradesh",
    "Moradabad": "Uttar Pradesh",
    "Gorakhpur": "Uttar Pradesh",

    # ───────────────── Madhya Pradesh ─────────────────
    "Indore": "Madhya Pradesh",
    "Bhopal": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh",
    "Jabalpur": "Madhya Pradesh",
    "Ujjain": "Madhya Pradesh",

    # ───────────────── West Bengal ─────────────────
    "Kolkata": "West Bengal",
    "Howrah": "West Bengal",
    "Durgapur": "West Bengal",
    "Siliguri": "West Bengal",
    "Asansol": "West Bengal",

    # ───────────────── Bihar ─────────────────
    "Patna": "Bihar",
    "Gaya": "Bihar",
    "Muzaffarpur": "Bihar",
    "Bhagalpur": "Bihar",

    # ───────────────── Punjab ─────────────────
    "Ludhiana": "Punjab",
    "Amritsar": "Punjab",
    "Jalandhar": "Punjab",
    "Patiala": "Punjab",
    "Mohali": "Punjab",

    # ───────────────── Haryana ─────────────────
    "Faridabad": "Haryana",
    "Gurgaon": "Haryana",
    "Panipat": "Haryana",
    "Ambala": "Haryana",
    "Hisar": "Haryana",

    # ───────────────── Kerala ─────────────────
    "Kochi": "Kerala",
    "Thiruvananthapuram": "Kerala",
    "Kozhikode": "Kerala",
    "Thrissur": "Kerala",
    "Kannur": "Kerala",

    # ───────────────── Odisha ─────────────────
    "Bhubaneswar": "Odisha",
    "Cuttack": "Odisha",
    "Rourkela": "Odisha",
    "Sambalpur": "Odisha",

    # ───────────────── Assam ─────────────────
    "Guwahati": "Assam",
    "Dibrugarh": "Assam",
    "Silchar": "Assam",

    # ───────────────── Jharkhand ─────────────────
    "Ranchi": "Jharkhand",
    "Jamshedpur": "Jharkhand",
    "Dhanbad": "Jharkhand",

    # ───────────────── Chhattisgarh ─────────────────
    "Raipur": "Chhattisgarh",
    "Bilaspur": "Chhattisgarh",
    "Durg": "Chhattisgarh",

    # ───────────────── Himachal Pradesh ─────────────────
    "Shimla": "Himachal Pradesh",
    "Dharamshala": "Himachal Pradesh",
    "Solan": "Himachal Pradesh",

    # ───────────────── Uttarakhand ─────────────────
    "Dehradun": "Uttarakhand",
    "Haridwar": "Uttarakhand",
    "Roorkee": "Uttarakhand",

    # ───────────────── Goa ─────────────────
    "Panaji": "Goa",
    "Margao": "Goa",
    "Vasco da Gama": "Goa",

    # ───────────────── Jammu & Kashmir ─────────────────
    "Srinagar": "Jammu and Kashmir",
    "Jammu": "Jammu and Kashmir",

    # ───────────────── Chandigarh ─────────────────
    "Chandigarh": "Chandigarh",

    # ───────────────── Tripura ─────────────────
    "Agartala": "Tripura",

    # ───────────────── Meghalaya ─────────────────
    "Shillong": "Meghalaya",

    # ───────────────── Manipur ─────────────────
    "Imphal": "Manipur",

    # ───────────────── Nagaland ─────────────────
    "Kohima": "Nagaland",

    # ───────────────── Mizoram ─────────────────
    "Aizawl": "Mizoram",

    # ───────────────── Arunachal Pradesh ─────────────────
    "Itanagar": "Arunachal Pradesh",

    # ───────────────── Sikkim ─────────────────
    "Gangtok": "Sikkim",
}

# ── City tiers for realistic sampling ─────────────────────────────

TIER_1_CITIES = {
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Pune",
    "Ahmedabad",
}

TIER_2_CITIES = {
    "Jaipur",
    "Lucknow",
    "Kanpur",
    "Nagpur",
    "Indore",
    "Bhopal",
    "Visakhapatnam",
    "Patna",
    "Vadodara",
    "Ghaziabad",
    "Ludhiana",
    "Agra",
    "Nashik",
    "Faridabad",
    "Meerut",
    "Rajkot",
    "Varanasi",
    "Srinagar",
    "Aurangabad",
    "Dhanbad",
    "Noida",
    "Surat",
    "Coimbatore",
    "Kochi",
    "Chandigarh",
    "Mysore",
}

ALL_CITIES = list(CITY_STATE_MAP.keys())

# Dynamic weighting:
# Tier 1 cities appear more frequently
CITY_WEIGHTS = []

for city in ALL_CITIES:
    if city in TIER_1_CITIES:
        CITY_WEIGHTS.append(4.0)
    elif city in TIER_2_CITIES:
        CITY_WEIGHTS.append(2.0)
    else:
        CITY_WEIGHTS.append(1.0)

CITY_WEIGHTS = (
    np.array(CITY_WEIGHTS, dtype=float) /
    np.sum(CITY_WEIGHTS)
).tolist()

CITY_METADATA = {
    city: {
        "tier": (
            "tier1" if city in TIER_1_CITIES
            else "tier2" if city in TIER_2_CITIES
            else "tier3"
        ),
        "state": state,
        "urbanity": (
            0.95 if city in TIER_1_CITIES
            else 0.72 if city in TIER_2_CITIES
            else 0.48
        ),
        "tech_hub": city in {"Bangalore", "Hyderabad", "Pune", "Noida", "Gurgaon", "Chennai"},
        "education_hub": city in {"Delhi", "New Delhi", "Bangalore", "Chennai", "Kolkata", "Pune", "Hyderabad"},
    }
    for city, state in CITY_STATE_MAP.items()
}

CITY_METADATA.update({
    "Bangalore": {
        "tier": "tier1",
        "state": "Karnataka",
        "urbanity": 0.95,
        "tech_hub": True,
        "education_hub": True,
    },
    "Mumbai": {
        "tier": "tier1",
        "state": "Maharashtra",
        "urbanity": 0.98,
        "tech_hub": False,
        "education_hub": True,
    },
    "Delhi": {
        "tier": "tier1",
        "state": "Delhi",
        "urbanity": 0.97,
        "tech_hub": False,
        "education_hub": True,
    },
    "Hyderabad": {
        "tier": "tier1",
        "state": "Telangana",
        "urbanity": 0.94,
        "tech_hub": True,
        "education_hub": True,
    },
})

CITY_METADATA.update({
    f"Rural {state}": {
        "tier": "rural",
        "state": state,
        "urbanity": 0.18,
        "tech_hub": False,
        "education_hub": False,
    }
    for state in sorted(set(CITY_STATE_MAP.values()))
})


COUNTRIES = ["India", "United States", "United Kingdom", "Canada", "Australia",
             "UAE", "Singapore", "Germany", "New Zealand"]
COUNTRY_WEIGHTS = [0.78, 0.07, 0.04, 0.03, 0.03, 0.03, 0.01, 0.005, 0.005]

# ── Income model ──────────────────────────────────────────────────────────────
# income_lpa = base + experience_bonus + field_multiplier + noise
# These are realistic for Indian job market (2024 values in LPA)

# Regional demographic covariance priors

REGIONAL_PRIORS = {
    "West Bengal": {
        "state_weight": 0.10,
        "religions": {
            "Hindu": 0.74,
            "Muslim": 0.24,
            "Christian": 0.01,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Bengali": 0.82,
            "Hindi": 0.10,
            "Urdu": 0.05,
            "English": 0.03,
        },
        "name_style": "bengali",
    },
    "Maharashtra": {
        "state_weight": 0.15,
        "religions": {
            "Hindu": 0.79,
            "Muslim": 0.12,
            "Buddhist": 0.06,
            "Jain": 0.01,
            "Christian": 0.01,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Marathi": 0.68,
            "Hindi": 0.18,
            "Urdu": 0.06,
            "Gujarati": 0.04,
            "English": 0.04,
        },
        "name_style": "marathi",
    },
    "Delhi": {
        "state_weight": 0.07,
        "religions": {
            "Hindu": 0.82,
            "Muslim": 0.13,
            "Sikh": 0.03,
            "Christian": 0.01,
            "Jain": 0.01,
        },
        "mother_tongues": {
            "Hindi": 0.74,
            "Punjabi": 0.10,
            "Urdu": 0.08,
            "English": 0.08,
        },
        "name_style": "north_indian",
    },
    "Karnataka": {
        "state_weight": 0.10,
        "religions": {
            "Hindu": 0.84,
            "Muslim": 0.13,
            "Christian": 0.02,
            "Jain": 0.005,
            "Other": 0.005,
        },
        "mother_tongues": {
            "Kannada": 0.66,
            "Urdu": 0.10,
            "Telugu": 0.08,
            "Tamil": 0.06,
            "Hindi": 0.05,
            "English": 0.05,
        },
        "name_style": "kannada",
    },
    "Tamil Nadu": {
        "state_weight": 0.10,
        "religions": {
            "Hindu": 0.88,
            "Christian": 0.06,
            "Muslim": 0.05,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Tamil": 0.88,
            "Telugu": 0.05,
            "Urdu": 0.03,
            "Malayalam": 0.02,
            "English": 0.02,
        },
        "name_style": "tamil",
    },
    "Telangana": {
        "state_weight": 0.08,
        "religions": {
            "Hindu": 0.85,
            "Muslim": 0.13,
            "Christian": 0.01,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Telugu": 0.73,
            "Urdu": 0.12,
            "Hindi": 0.06,
            "Kannada": 0.03,
            "English": 0.06,
        },
        "name_style": "telugu",
    },
    "Gujarat": {
        "state_weight": 0.09,
        "religions": {
            "Hindu": 0.88,
            "Muslim": 0.10,
            "Jain": 0.01,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Gujarati": 0.82,
            "Hindi": 0.10,
            "Urdu": 0.04,
            "English": 0.04,
        },
        "name_style": "gujarati",
    },
    "Punjab": {
        "state_weight": 0.06,
        "religions": {
            "Sikh": 0.58,
            "Hindu": 0.38,
            "Muslim": 0.02,
            "Christian": 0.01,
            "Other": 0.01,
        },
        "mother_tongues": {
            "Punjabi": 0.86,
            "Hindi": 0.08,
            "Urdu": 0.02,
            "English": 0.04,
        },
        "name_style": "punjabi",
    },
    "Kerala": {
        "state_weight": 0.07,
        "religions": {
            "Hindu": 0.55,
            "Muslim": 0.27,
            "Christian": 0.18,
        },
        "mother_tongues": {
            "Malayalam": 0.92,
            "Tamil": 0.03,
            "Hindi": 0.02,
            "English": 0.03,
        },
        "name_style": "malayali",
    },
    "Uttar Pradesh": {
        "state_weight": 0.18,
        "religions": {
            "Hindu": 0.80,
            "Muslim": 0.19,
            "Christian": 0.005,
            "Other": 0.005,
        },
        "mother_tongues": {
            "Hindi": 0.82,
            "Urdu": 0.12,
            "Bhojpuri": 0.03,
            "English": 0.03,
        },
        "name_style": "north_indian",
    },
}

CASTE_PRIORS_BY_RELIGION = {
    "Hindu": {
        "Brahmin": 0.12,
        "Kshatriya": 0.10,
        "Vaishya": 0.08,
        "OBC": 0.34,
        "SC": 0.18,
        "ST": 0.08,
        "Not specified": 0.10,
    },
    "Muslim": {
        "Sunni": 0.78,
        "Shia": 0.12,
        "Ahmadiyya": 0.01,
        "Not specified": 0.09,
    },
    "Sikh": {
        "Jat": 0.46,
        "Khatri": 0.22,
        "Arora": 0.18,
        "Not specified": 0.14,
    },
    "Christian": {
        "Catholic": 0.42,
        "Protestant": 0.33,
        "Orthodox": 0.15,
        "Not specified": 0.10,
    },
    "Jain": {
        "Shvetambara": 0.48,
        "Digambara": 0.48,
        "Not specified": 0.04,
    },
    "Buddhist": {
        "Mahayana": 0.42,
        "Theravada": 0.18,
        "Not specified": 0.40,
    },
    "Parsi": {
        "Parsi": 0.90,
        "Not specified": 0.10,
    },
    "Other": {
        "Not specified": 1.0,
    },
}

CASTES_BY_STATE_AND_RELIGION = {
    "West Bengal": {
        "Hindu": {
            "Brahmin": 0.18,
            "Kshatriya": 0.04,
            "Vaishya": 0.06,
            "OBC": 0.32,
            "SC": 0.24,
            "ST": 0.07,
            "Not specified": 0.09,
        },
        "Muslim": {
            "Sunni": 0.84,
            "Shia": 0.04,
            "Ahmadiyya": 0.005,
            "Not specified": 0.115,
        },
        "Christian": {
            "Catholic": 0.34,
            "Protestant": 0.42,
            "Orthodox": 0.04,
            "Not specified": 0.20,
        },
        "Other": {"Not specified": 1.0},
    },
    "Maharashtra": {
        "Hindu": {
            "Brahmin": 0.10,
            "Kshatriya": 0.08,
            "Vaishya": 0.07,
            "OBC": 0.38,
            "SC": 0.16,
            "ST": 0.10,
            "Not specified": 0.11,
        },
        "Muslim": {
            "Sunni": 0.80,
            "Shia": 0.08,
            "Ahmadiyya": 0.005,
            "Not specified": 0.115,
        },
        "Buddhist": {
            "Mahayana": 0.30,
            "Theravada": 0.10,
            "Not specified": 0.60,
        },
        "Jain": {
            "Shvetambara": 0.62,
            "Digambara": 0.24,
            "Not specified": 0.14,
        },
        "Christian": {
            "Catholic": 0.54,
            "Protestant": 0.24,
            "Orthodox": 0.08,
            "Not specified": 0.14,
        },
        "Other": {"Not specified": 1.0},
    },
    "Delhi": {
        "Hindu": {
            "Brahmin": 0.13,
            "Kshatriya": 0.13,
            "Vaishya": 0.13,
            "OBC": 0.30,
            "SC": 0.17,
            "ST": 0.02,
            "Not specified": 0.12,
        },
        "Muslim": {
            "Sunni": 0.76,
            "Shia": 0.12,
            "Ahmadiyya": 0.005,
            "Not specified": 0.115,
        },
        "Sikh": {
            "Khatri": 0.32,
            "Arora": 0.30,
            "Jat": 0.18,
            "Not specified": 0.20,
        },
        "Christian": {
            "Catholic": 0.35,
            "Protestant": 0.40,
            "Orthodox": 0.08,
            "Not specified": 0.17,
        },
        "Other": {"Not specified": 1.0},
    },
    "Karnataka": {
        "Hindu": {
            "Brahmin": 0.07,
            "Kshatriya": 0.05,
            "Vaishya": 0.07,
            "OBC": 0.42,
            "SC": 0.17,
            "ST": 0.08,
            "Not specified": 0.14,
        },
        "Muslim": {
            "Sunni": 0.78,
            "Shia": 0.08,
            "Ahmadiyya": 0.005,
            "Not specified": 0.135,
        },
        "Christian": {
            "Catholic": 0.44,
            "Protestant": 0.35,
            "Orthodox": 0.06,
            "Not specified": 0.15,
        },
        "Jain": {
            "Digambara": 0.52,
            "Shvetambara": 0.32,
            "Not specified": 0.16,
        },
        "Other": {"Not specified": 1.0},
    },
    "Tamil Nadu": {
        "Hindu": {
            "Brahmin": 0.04,
            "Kshatriya": 0.04,
            "Vaishya": 0.06,
            "OBC": 0.52,
            "SC": 0.20,
            "ST": 0.01,
            "Not specified": 0.13,
        },
        "Muslim": {
            "Sunni": 0.82,
            "Shia": 0.05,
            "Ahmadiyya": 0.005,
            "Not specified": 0.125,
        },
        "Christian": {
            "Catholic": 0.50,
            "Protestant": 0.32,
            "Orthodox": 0.05,
            "Not specified": 0.13,
        },
        "Other": {"Not specified": 1.0},
    },
    "Telangana": {
        "Hindu": {
            "Brahmin": 0.06,
            "Kshatriya": 0.08,
            "Vaishya": 0.06,
            "OBC": 0.43,
            "SC": 0.16,
            "ST": 0.09,
            "Not specified": 0.12,
        },
        "Muslim": {
            "Sunni": 0.74,
            "Shia": 0.14,
            "Ahmadiyya": 0.005,
            "Not specified": 0.115,
        },
        "Christian": {
            "Catholic": 0.32,
            "Protestant": 0.46,
            "Orthodox": 0.06,
            "Not specified": 0.16,
        },
        "Other": {"Not specified": 1.0},
    },
    "Gujarat": {
        "Hindu": {
            "Brahmin": 0.09,
            "Kshatriya": 0.13,
            "Vaishya": 0.14,
            "OBC": 0.34,
            "SC": 0.08,
            "ST": 0.12,
            "Not specified": 0.10,
        },
        "Muslim": {
            "Sunni": 0.76,
            "Shia": 0.10,
            "Ahmadiyya": 0.005,
            "Not specified": 0.135,
        },
        "Jain": {
            "Shvetambara": 0.70,
            "Digambara": 0.16,
            "Not specified": 0.14,
        },
        "Other": {"Not specified": 1.0},
    },
    "Punjab": {
        "Sikh": {
            "Jat": 0.52,
            "Khatri": 0.16,
            "Arora": 0.14,
            "Not specified": 0.18,
        },
        "Hindu": {
            "Brahmin": 0.10,
            "Kshatriya": 0.08,
            "Vaishya": 0.16,
            "OBC": 0.32,
            "SC": 0.22,
            "ST": 0.01,
            "Not specified": 0.11,
        },
        "Muslim": {
            "Sunni": 0.82,
            "Shia": 0.06,
            "Ahmadiyya": 0.005,
            "Not specified": 0.115,
        },
        "Christian": {
            "Catholic": 0.22,
            "Protestant": 0.52,
            "Orthodox": 0.04,
            "Not specified": 0.22,
        },
        "Other": {"Not specified": 1.0},
    },
    "Kerala": {
        "Hindu": {
            "Brahmin": 0.03,
            "Kshatriya": 0.04,
            "Vaishya": 0.05,
            "OBC": 0.46,
            "SC": 0.10,
            "ST": 0.02,
            "Not specified": 0.30,
        },
        "Muslim": {
            "Sunni": 0.86,
            "Shia": 0.03,
            "Ahmadiyya": 0.005,
            "Not specified": 0.105,
        },
        "Christian": {
            "Catholic": 0.40,
            "Protestant": 0.22,
            "Orthodox": 0.28,
            "Not specified": 0.10,
        },
    },
    "Uttar Pradesh": {
        "Hindu": {
            "Brahmin": 0.12,
            "Kshatriya": 0.10,
            "Vaishya": 0.08,
            "OBC": 0.40,
            "SC": 0.20,
            "ST": 0.01,
            "Not specified": 0.09,
        },
        "Muslim": {
            "Sunni": 0.78,
            "Shia": 0.12,
            "Ahmadiyya": 0.005,
            "Not specified": 0.095,
        },
        "Christian": {
            "Catholic": 0.26,
            "Protestant": 0.48,
            "Orthodox": 0.06,
            "Not specified": 0.20,
        },
        "Other": {"Not specified": 1.0},
    },
}

MIGRANT_MOTHER_TONGUE_DIST = {
    "Hindi": 0.35,
    "English": 0.20,
    "Urdu": 0.10,
    "Bengali": 0.06,
    "Marathi": 0.06,
    "Tamil": 0.06,
    "Telugu": 0.06,
    "Kannada": 0.05,
    "Malayalam": 0.03,
    "Punjabi": 0.03,
}

NAMES_BY_STYLE = {
    "bengali": {
        "Male": ["Soumik", "Arindam", "Subhajit", "Sourav", "Anirban", "Sagnik", "Debjit", "Ritwik"],
        "Female": ["Madhumita", "Sohini", "Ananya", "Ritwika", "Debolina", "Moumita", "Sanchita", "Ishita"],
        "Surnames": ["Ghosh", "Das", "Dutta", "Sengupta", "Roy", "Pal"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Chatterjee", "Mukherjee", "Banerjee", "Bhattacharya", "Ganguly"],
            "Kshatriya": ["Roy", "Chowdhury", "Sengupta"],
            "Vaishya": ["Dutta", "Saha", "Banik"],
            "OBC": ["Das", "Mondal", "Pal"],
            "SC": ["Das", "Naskar", "Barman"],
            "ST": ["Murmu", "Soren", "Tudu"],
        },
        "Muslim": {
            "Male": ["Ayaan", "Rehan", "Imran", "Sajid", "Farhan", "Arif"],
            "Female": ["Nusrat", "Sania", "Farzana", "Ayesha", "Tasnim", "Ruksana"],
            "Surnames": ["Rahman", "Islam", "Ali", "Hossain", "Khan"],
        },
    },
    "marathi": {
        "Male": ["Saurabh", "Prathamesh", "Nikhil", "Omkar", "Akshay", "Mandar", "Rohan", "Sameer"],
        "Female": ["Aditi", "Pooja", "Sneha", "Mrunal", "Prajakta", "Neha", "Gauri", "Manasi"],
        "Surnames": ["Patil", "Kale", "Shinde", "More", "Jadhav", "Pawar"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Kulkarni", "Joshi", "Gokhale", "Apte", "Deshpande"],
            "Kshatriya": ["Deshmukh", "Bhosale", "Chavan"],
            "Vaishya": ["Shah", "Lodha", "Kothari"],
            "OBC": ["Patil", "Shinde", "Jadhav", "Pawar"],
            "SC": ["Kamble", "Gaikwad", "Jadhav"],
            "ST": ["Pawar", "Kokate", "Madavi"],
        },
        "Muslim": {
            "Male": ["Aamir", "Sameer", "Irfan", "Faizan", "Nadeem", "Zeeshan"],
            "Female": ["Sana", "Ayesha", "Farah", "Nazia", "Alfiya", "Zoya"],
            "Surnames": ["Shaikh", "Pathan", "Khan", "Qureshi", "Ansari"],
        },
    },
    "north_indian": {
        "Male": ["Amit", "Rahul", "Ankit", "Gaurav", "Abhishek", "Sandeep", "Rohit", "Vivek"],
        "Female": ["Priya", "Neha", "Pooja", "Anjali", "Shreya", "Ritu", "Nidhi", "Sakshi"],
        "Surnames": ["Verma", "Singh", "Srivastava", "Yadav", "Chauhan", "Tiwari"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Sharma", "Mishra", "Tiwari", "Pandey", "Tripathi"],
            "Kshatriya": ["Singh", "Chauhan", "Rathore", "Tomar"],
            "Vaishya": ["Gupta", "Agarwal", "Jain", "Bansal"],
            "OBC": ["Yadav", "Verma", "Kushwaha", "Maurya"],
            "SC": ["Paswan", "Jatav", "Valmiki", "Kumar"],
            "ST": ["Gond", "Kharwar", "Kol"],
        },
        "Muslim": {
            "Male": ["Ayaan", "Faizan", "Danish", "Armaan", "Sahil", "Adnan"],
            "Female": ["Ayesha", "Sana", "Alisha", "Zoya", "Nazia", "Saba"],
            "Surnames": ["Khan", "Ansari", "Qureshi", "Siddiqui", "Rizvi"],
        },
        "Sikh": {
            "Male": ["Gurpreet", "Harpreet", "Jaspreet", "Manpreet", "Amrit", "Paramjit"],
            "Female": ["Gurleen", "Harleen", "Jasleen", "Simran", "Amandeep", "Navdeep"],
            "Surnames": ["Singh", "Kaur"],
        },
    },
    "kannada": {
        "Male": ["Naveen", "Raghavendra", "Kiran", "Pradeep", "Chethan", "Karthik", "Srinivas", "Darshan"],
        "Female": ["Anusha", "Divya", "Pavithra", "Shruthi", "Kavya", "Spoorthi", "Meghana", "Deepika"],
        "Surnames": ["Rao", "Gowda", "Shetty", "Murthy", "Acharya", "Naik"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Rao", "Bhat", "Hegde", "Kulkarni"],
            "Kshatriya": ["Urs", "Arasu", "Wodeyar"],
            "Vaishya": ["Shetty", "Setty", "Pai"],
            "OBC": ["Gowda", "Naik", "Poojary"],
            "SC": ["Madiga", "Chalavadi", "Holeya"],
            "ST": ["Naik", "Gowda", "Soliga"],
        },
        "Muslim": {
            "Male": ["Irfan", "Suhail", "Imran", "Azeem", "Nawaz", "Fayaz"],
            "Female": ["Saba", "Ayesha", "Farheen", "Nida", "Zainab", "Sumaiya"],
            "Surnames": ["Khan", "Pasha", "Sharif", "Sayeed", "Baig"],
        },
    },
    "tamil": {
        "Male": ["Arvind", "Arun", "Karthik", "Vignesh", "Suresh", "Prakash", "Saravanan", "Balaji"],
        "Female": ["Divya", "Meena", "Priya", "Aishwarya", "Kavitha", "Nandhini", "Revathi", "Lakshmi"],
        "Surnames": ["Raman", "Subramanian", "Krishnan", "Ramasamy", "Natarajan", "Srinivasan"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Iyer", "Iyengar", "Subramanian", "Srinivasan"],
            "Kshatriya": ["Rajan", "Varma", "Thevar"],
            "Vaishya": ["Chettiar", "Mudaliar", "Pillai"],
            "OBC": ["Ramasamy", "Natarajan", "Gounder"],
            "SC": ["Paraiyar", "Pallan", "Kumar"],
            "ST": ["Malayali", "Irular", "Kurumbar"],
        },
        "Muslim": {
            "Male": ["Imran", "Faisal", "Rizwan", "Niyas", "Shahul", "Ameer"],
            "Female": ["Fathima", "Ayesha", "Nisha", "Rizwana", "Saira", "Haseena"],
            "Surnames": ["Rowther", "Maraicar", "Labbai", "Khan", "Syed"],
        },
    },
    "telugu": {
        "Male": ["Srinivas", "Venkatesh", "Sai", "Ravi", "Chaitanya", "Kiran", "Praneeth", "Nikhil"],
        "Female": ["Sravani", "Harika", "Deepthi", "Anusha", "Kavya", "Mounika", "Lasya", "Sindhu"],
        "Surnames": ["Rao", "Naidu", "Goud", "Prasad", "Murthy", "Yadav"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Sastry", "Sharma", "Murthy", "Rao"],
            "Kshatriya": ["Varma", "Raju", "Naidu"],
            "Vaishya": ["Gupta", "Setty", "Komati"],
            "OBC": ["Goud", "Yadav", "Naidu"],
            "SC": ["Mala", "Madiga", "Dasari"],
            "ST": ["Naik", "Koya", "Gond"],
        },
        "Muslim": {
            "Male": ["Azeem", "Sohail", "Naveed", "Imran", "Farhan", "Yusuf"],
            "Female": ["Sameera", "Ayesha", "Sana", "Farheen", "Nazia", "Zainab"],
            "Surnames": ["Khan", "Syed", "Shaik", "Pasha", "Qureshi"],
        },
    },
    "gujarati": {
        "Male": ["Nirav", "Bhavesh", "Jignesh", "Ketan", "Hardik", "Parth", "Chirag", "Mehul"],
        "Female": ["Hetal", "Kinjal", "Pooja", "Nisha", "Riddhi", "Jinal", "Dhwani", "Ami"],
        "Surnames": ["Patel", "Shah", "Mehta", "Desai", "Thakkar", "Vyas"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Trivedi", "Joshi", "Vyas", "Dave"],
            "Kshatriya": ["Jadeja", "Solanki", "Rathod"],
            "Vaishya": ["Shah", "Mehta", "Thakkar", "Doshi"],
            "OBC": ["Patel", "Prajapati", "Modi"],
            "SC": ["Solanki", "Makwana", "Parmar"],
            "ST": ["Vasava", "Bhil", "Rathwa"],
        },
        "Muslim": {
            "Male": ["Imran", "Aadil", "Sohail", "Arif", "Junaid", "Firoz"],
            "Female": ["Ayesha", "Nafisa", "Sana", "Farida", "Zoya", "Hina"],
            "Surnames": ["Memon", "Vohra", "Khan", "Shaikh", "Saiyed"],
        },
    },
    "punjabi": {
        "Male": ["Karan", "Sandeep", "Rohit", "Aman", "Vikas", "Nitin", "Rajiv", "Deepak"],
        "Female": ["Simran", "Kiran", "Pooja", "Neha", "Ritu", "Sonia", "Mehak", "Rupinder"],
        "Surnames": ["Arora", "Bedi", "Chopra", "Malhotra", "Khanna", "Kapoor"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Sharma", "Bhardwaj", "Dutt"],
            "Kshatriya": ["Rajput", "Rana", "Chauhan"],
            "Vaishya": ["Arora", "Bansal", "Gupta"],
            "OBC": ["Saini", "Kamboj", "Dhiman"],
            "SC": ["Ram", "Lal", "Masiih"],
            "ST": ["Not specified"],
        },
        "Muslim": {
            "Male": ["Adeel", "Imran", "Nadeem", "Sajjad", "Usman", "Bilal"],
            "Female": ["Sadia", "Ayesha", "Hina", "Nazia", "Saira", "Noor"],
            "Surnames": ["Khan", "Sheikh", "Butt", "Malik", "Qureshi"],
        },
        "Sikh": {
            "Male": ["Gurpreet", "Harpreet", "Jaspreet", "Manpreet", "Amrit", "Paramjit"],
            "Female": ["Gurleen", "Harleen", "Jasleen", "Simran", "Amandeep", "Navdeep"],
            "Surnames": ["Singh", "Kaur"],
        },
    },
    "malayali": {
        "Male": ["Arun", "Nithin", "Vishnu", "Rahul", "Sandeep", "Akhil", "Anand", "Krishnan"],
        "Female": ["Anju", "Aparna", "Meera", "Nimisha", "Sneha", "Devika", "Anjana", "Lakshmi"],
        "Surnames": ["Nair", "Menon", "Pillai", "Panicker", "Warrier", "Kurup"],
        "HinduSurnamesByCaste": {
            "Brahmin": ["Namboodiri", "Iyer", "Menon"],
            "Kshatriya": ["Varma", "Thampuran", "Raja"],
            "Vaishya": ["Menon", "Pillai", "Nair"],
            "OBC": ["Ezhava", "Nair", "Panicker"],
            "SC": ["Pulayan", "Cheramar", "Kumar"],
            "ST": ["Kurichiya", "Paniya", "Adiya"],
        },
        "Muslim": {
            "Male": ["Niyas", "Shafi", "Faisal", "Rasheed", "Noushad", "Afsal"],
            "Female": ["Fathima", "Aysha", "Shabana", "Nazeera", "Rinsha", "Suhana"],
            "Surnames": ["Mappila", "Kunhi", "Koya", "Rahman", "Ali"],
        },
        "Christian": {
            "Male": ["Jobin", "Mathew", "Thomas", "Albin", "George", "Jose"],
            "Female": ["Ann", "Maria", "Merin", "Ancy", "Rose", "Tresa"],
            "Surnames": ["Kurian", "Varghese", "Thomas", "Joseph", "Mathew"],
        },
    },
}

# Migration is sampled per profile from this range so runs remain probabilistic
# while staying near the requested 15-25% band.
MIGRATION_PROBABILITY_RANGE = (0.15, 0.25)


INCOME_BASE_BY_PROFESSION = {
    "Software Engineer":        {"base": 6.0,  "per_year_exp": 1.2,  "sigma": 2.0},
    "Senior Software Engineer": {"base": 14.0, "per_year_exp": 1.8,  "sigma": 3.0},
    "Product Manager":          {"base": 18.0, "per_year_exp": 2.0,  "sigma": 4.0},
    "Data Scientist":           {"base": 12.0, "per_year_exp": 1.5,  "sigma": 2.5},
    "Doctor":                   {"base": 8.0,  "per_year_exp": 2.0,  "sigma": 3.0},
    "Professor":                {"base": 8.0,  "per_year_exp": 0.5,  "sigma": 1.5},
    "Teacher":                  {"base": 3.0,  "per_year_exp": 0.3,  "sigma": 0.8},
    "Lawyer":                   {"base": 6.0,  "per_year_exp": 1.5,  "sigma": 3.0},
    "Chartered Accountant":     {"base": 7.0,  "per_year_exp": 1.2,  "sigma": 2.0},
    "Bank Officer":             {"base": 5.0,  "per_year_exp": 0.8,  "sigma": 1.0},
    "Government Employee":      {"base": 5.0,  "per_year_exp": 0.5,  "sigma": 0.8},
    "Business Owner":           {"base": 8.0,  "per_year_exp": 1.0,  "sigma": 5.0},
    "Shop Owner":               {"base": 3.0,  "per_year_exp": 0.3,  "sigma": 1.0},
    "DEFAULT":                  {"base": 4.0,  "per_year_exp": 0.5,  "sigma": 1.5},
}

# ── Hobbies ───────────────────────────────────────────────────────────────────

# Hobby generation uses weighted clusters instead of uniform sampling. This
# preserves common matrimonial-profile cliches while letting age, city, income,
# education, and profession create realistic co-occurrence patterns.
COMMON_HOBBIES = {
    "Listening to music": 0.20,
    "Watching movies": 0.17,
    "Cooking": 0.16,
    "Spending time with family": 0.18,
    "Travelling": 0.15,
    "Cricket watching": 0.08,
    "Reading": 0.06,
}

MODERATELY_COMMON_HOBBIES = {
    "Gym": 0.10,
    "Yoga": 0.09,
    "Photography": 0.09,
    "Dancing": 0.08,
    "Singing": 0.07,
    "Gardening": 0.07,
    "Football": 0.06,
    "Badminton": 0.06,
    "Chess": 0.05,
    "Writing": 0.05,
    "Food blogging": 0.05,
    "Cafe hopping": 0.04,
    "Meditation": 0.04,
    "Volunteering": 0.04,
    "Cycling": 0.04,
    "Painting": 0.04,
    "Stock market investing": 0.03,
}

NICHE_HOBBIES = {
    "Trekking": 0.10,
    "Swimming": 0.08,
    "Classical dance": 0.08,
    "Devotional music": 0.08,
    "Temple visits": 0.07,
    "K-drama": 0.07,
    "Anime": 0.07,
    "Cricket analytics": 0.06,
    "Saree designing": 0.06,
    "Research reading": 0.06,
    "Podcast listening": 0.06,
    "Theatre": 0.05,
    "Running": 0.05,
    "Investing": 0.05,
    "Board games": 0.04,
    "Spiritual discourses": 0.04,
}

RARE_HOBBIES = {
    "Birdwatching": 0.12,
    "Meditation retreats": 0.11,
    "Astrology": 0.10,
    "Marathon training": 0.09,
    "Hindustani classical music": 0.08,
    "Carnatic music": 0.08,
    "Pottery": 0.07,
    "Calligraphy": 0.07,
    "Wildlife photography": 0.07,
    "Community theatre": 0.06,
    "Language learning": 0.05,
    "Heritage walks": 0.05,
    "Book collecting": 0.05,
}

HOBBY_TIER_WEIGHTS = {
    "common": 0.50,
    "moderate": 0.30,
    "niche": 0.15,
    "rare": 0.05,
}

HOBBY_CLUSTERS = {
    "urban_young": {
        "hobbies": {
            "Gaming": 0.17,
            "Listening to music": 0.15,
            "Gym": 0.16,
            "Watching movies": 0.14,
            "Football": 0.10,
            "Travelling": 0.12,
            "Cafe hopping": 0.08,
            "K-drama": 0.04,
            "Anime": 0.04,
        },
        "affinity": {"age_max": 32, "city_tiers": ["tier1", "tier2"]},
    },
    "traditional_family": {
        "hobbies": {
            "Cooking": 0.18,
            "Spending time with family": 0.20,
            "Listening to music": 0.12,
            "Watching movies": 0.10,
            "Gardening": 0.10,
            "Temple visits": 0.09,
            "Devotional music": 0.08,
            "Spiritual discourses": 0.05,
            "Astrology": 0.03,
            "Saree designing": 0.05,
        },
        "affinity": {"age_min": 30, "marital_statuses": ["divorced", "widowed", "separated"]},
    },
    "outdoors": {
        "hobbies": {
            "Travelling": 0.17,
            "Trekking": 0.15,
            "Photography": 0.14,
            "Cycling": 0.11,
            "Running": 0.08,
            "Swimming": 0.08,
            "Badminton": 0.07,
            "Wildlife photography": 0.06,
            "Birdwatching": 0.04,
            "Heritage walks": 0.05,
            "Cricket watching": 0.05,
        },
        "affinity": {"income_min": 8.0, "city_tiers": ["tier1", "tier2"]},
    },
    "intellectual": {
        "hobbies": {
            "Reading": 0.18,
            "Writing": 0.13,
            "Chess": 0.12,
            "Research reading": 0.11,
            "Podcast listening": 0.09,
            "Language learning": 0.07,
            "Book collecting": 0.06,
            "Volunteering": 0.06,
            "Meditation": 0.06,
            "Board games": 0.05,
            "Cricket analytics": 0.04,
            "Stock market investing": 0.03,
        },
        "affinity": {"education_levels": ["Post Graduate", "PhD"]},
    },
    "creative": {
        "hobbies": {
            "Photography": 0.14,
            "Painting": 0.12,
            "Dancing": 0.12,
            "Singing": 0.10,
            "Classical dance": 0.08,
            "Food blogging": 0.08,
            "Theatre": 0.07,
            "Pottery": 0.05,
            "Calligraphy": 0.05,
            "Community theatre": 0.04,
            "Listening to music": 0.10,
            "Cooking": 0.05,
        },
        "affinity": {"professions": ["Teacher", "Professor", "Marketing Executive", "HR Executive"]},
    },
    "fitness": {
        "hobbies": {
            "Gym": 0.18,
            "Yoga": 0.15,
            "Running": 0.12,
            "Cycling": 0.10,
            "Football": 0.09,
            "Badminton": 0.09,
            "Swimming": 0.08,
            "Meditation": 0.06,
            "Marathon training": 0.04,
            "Trekking": 0.05,
            "Cricket watching": 0.04,
        },
        "affinity": {"age_max": 40},
    },
    "regional_cultural": {
        "hobbies": {
            "Devotional music": 0.12,
            "Classical dance": 0.10,
            "Temple visits": 0.12,
            "Carnatic music": 0.07,
            "Hindustani classical music": 0.07,
            "Saree designing": 0.06,
            "Astrology": 0.05,
            "Spiritual discourses": 0.06,
            "Cooking": 0.12,
            "Spending time with family": 0.12,
            "Volunteering": 0.05,
            "Heritage walks": 0.06,
        },
        "affinity": {"age_min": 28},
    },
    "finance_status": {
        "hobbies": {
            "Stock market investing": 0.18,
            "Investing": 0.15,
            "Reading": 0.12,
            "Podcast listening": 0.10,
            "Travelling": 0.10,
            "Cafe hopping": 0.08,
            "Photography": 0.07,
            "Golf": 0.03,
            "Chess": 0.07,
            "Food blogging": 0.05,
            "Watching movies": 0.05,
        },
        "affinity": {"income_min": 15.0, "professions": ["Finance Manager", "Chartered Accountant", "Business Owner"]},
    },
    "quiet_home": {
        "hobbies": {
            "Listening to music": 0.16,
            "Watching movies": 0.15,
            "Reading": 0.13,
            "Cooking": 0.12,
            "Spending time with family": 0.14,
            "Gardening": 0.08,
            "Meditation": 0.07,
            "Yoga": 0.06,
            "K-drama": 0.04,
            "Singing": 0.05,
        },
        "affinity": {},
    },
}

HOBBY_COOCCURRENCE_GROUPS = {
    "active_sports": ["Gym", "Football", "Badminton", "Running", "Cycling", "Swimming", "Marathon training"],
    "outdoor_travel": ["Travelling", "Trekking", "Photography", "Wildlife photography", "Birdwatching", "Heritage walks"],
    "intellectual": ["Reading", "Writing", "Chess", "Research reading", "Podcast listening", "Book collecting", "Language learning"],
    "home_family": ["Cooking", "Spending time with family", "Gardening", "Watching movies", "Listening to music"],
    "creative": ["Painting", "Dancing", "Singing", "Classical dance", "Theatre", "Pottery", "Calligraphy", "Community theatre"],
    "cultural_spiritual": ["Devotional music", "Temple visits", "Spiritual discourses", "Astrology", "Carnatic music", "Hindustani classical music"],
    "internet_pop": ["Gaming", "Anime", "K-drama", "Cafe hopping", "Food blogging", "Watching movies"],
    "finance": ["Stock market investing", "Investing", "Podcast listening", "Chess"],
}

HOBBY_INCOMPATIBLE_GROUPS = {
    "cultural_spiritual": ["internet_pop"],
    "internet_pop": ["cultural_spiritual"],
    "active_sports": ["cultural_spiritual"],
}

# Backward-compatible flattened pool for older scripts. New generation should
# use hobby_generator.generate_hobbies(...).
HOBBIES_POOL = sorted({
    hobby
    for cluster in HOBBY_CLUSTERS.values()
    for hobby in cluster["hobbies"]
})
