"""
constants.py
All domain-specific lookup tables for Indian matrimonial profiles.
Keeping these separate means you can expand the vocabulary without
touching any generation logic.
"""

import numpy as np

# ── Demographic pools ─────────────────────────────────────────────────────────

RELIGIONS = ["Hindu", "Muslim", "Sikh", "Christian", "Jain", "Buddhist", "Parsi"]
RELIGION_WEIGHTS = [0.79, 0.12, 0.02, 0.02, 0.02, 0.01, 0.01]  # approx India census

CASTES_BY_RELIGION = {
    "Hindu":    ["Brahmin", "Kshatriya", "Vaishya", "OBC", "SC", "ST", "Not specified"],
    "Muslim":   ["Sunni", "Shia", "Ahmadiyya", "Not specified"],
    "Sikh":     ["Jat", "Khatri", "Arora", "Not specified"],
    "Christian":["Catholic", "Protestant", "Orthodox", "Not specified"],
    "Jain":     ["Digambara", "Shvetambara", "Not specified"],
    "Buddhist": ["Mahayana", "Theravada", "Not specified"],
    "Parsi":    ["Parsi", "Not specified"],
}

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
    "Post Graduate": ["MBA", "M.Tech", "M.Sc", "MA", "MCA", "LLM", "MD", "MS"],
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

COLLEGE_TIER_META = {
    "elite": {
        "acceptance_weight": 0.008,
        "avg_salary_multiplier": 2.8,
        "migration_affinity": 0.87,
    },
    "premium": {
        "acceptance_weight": 0.04,
        "avg_salary_multiplier": 2.0,
        "migration_affinity": 0.65,
    },
    "upper_mid": {
        "acceptance_weight": 0.14,
        "avg_salary_multiplier": 1.4,
        "migration_affinity": 0.44,
    },
    "mid": {
        "acceptance_weight": 0.30,
        "avg_salary_multiplier": 1.0,
        "migration_affinity": 0.24,
    },
    "local": {
        "acceptance_weight": 0.55,
        "avg_salary_multiplier": 0.72,
        "migration_affinity": 0.12,
    },
}

for level, categories in COLLEGES.items():
    for category, names in categories.items():
        for name in names:
            if category.startswith("elite") or name.startswith(("IIT", "IIM", "BITS", "AIIMS", "IISc", "TIFR")):
                tier = "elite"
            elif category in {"top_universities", "technical_pg", "medical_pg", "top_research", "good_universities"}:
                tier = "premium"
            elif category in {"private_reputed", "general_pg"}:
                tier = "upper_mid"
            elif category in {"regional_state", "generic"}:
                tier = "mid"
            else:
                tier = "local"

            tier_meta = COLLEGE_TIER_META[tier]
            fields = ["General"]
            if any(k in name for k in ["IIT", "NIT", "IIIT", "Engineering", "BITS"]):
                fields = ["Engineering", "Computer Science", "Science"]
            elif any(k in name for k in ["AIIMS", "CMC", "Medical"]):
                fields = ["Medicine", "Nursing", "Pharmacy"]
            elif any(k in name for k in ["IIM", "XLRI", "Management"]):
                fields = ["MBA", "Management", "Commerce"]
            elif any(k in name for k in ["University", "College", "JNU"]):
                fields = ["Arts", "Science", "Commerce", "Law"]

            field_multiplier = 1.0
            if "Medicine" in fields:
                field_multiplier = 1.12
            elif "Management" in fields:
                field_multiplier = 1.08
            elif "Computer Science" in fields:
                field_multiplier = 1.05

            COLLEGE_METADATA[name] = {
                "tier": tier,
                "fields": fields,
                "city": None,
                "category": category,
                "acceptance_weight": tier_meta["acceptance_weight"],
                "avg_salary_multiplier": tier_meta["avg_salary_multiplier"] * field_multiplier,
                "migration_affinity": tier_meta["migration_affinity"],
            }

COLLEGE_METADATA.update({
    "IIT Delhi": {
        "tier": "elite",
        "fields": ["Engineering", "Computer Science"],
        "city": "Delhi",
        "category": "elite_engineering",
        "acceptance_weight": 0.01,
    },
    "IIT Bombay": {
        "tier": "elite",
        "fields": ["Engineering", "Computer Science"],
        "city": "Mumbai",
        "category": "elite_engineering",
        "acceptance_weight": 0.01,
    },
    "IIT Madras": {
        "tier": "elite",
        "fields": ["Engineering", "Computer Science"],
        "city": "Chennai",
        "category": "elite_engineering",
        "acceptance_weight": 0.01,
    },
    "AIIMS Delhi": {
        "tier": "elite",
        "fields": ["Medicine"],
        "city": "Delhi",
        "category": "medical_pg",
        "acceptance_weight": 0.01,
        "avg_salary_multiplier": 2.4,
        "migration_affinity": 0.78,
    },
    "Delhi University": {
        "tier": "premium",
        "fields": ["Arts", "Science", "Commerce", "Law"],
        "city": "Delhi",
        "category": "top_universities",
        "acceptance_weight": 0.05,
        "avg_salary_multiplier": 1.45,
        "migration_affinity": 0.62,
    },
})

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
                 "Apple India", "Flipkart", "Zomato", "Swiggy", "Paytm", "Ola", "Uber India", "Linkedin", "Adobe India", "Salesforce India", "Oracle India", "SAP India",
                 "Samsung"],
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
    "Google": ["google.com"],
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
