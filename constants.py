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

# Realistic colleges: mix of real and plausible fabricated names
COLLEGES = {
    "10th":          ["Local Government School", "St. Xavier's School", "Kendriya Vidyalaya"],
    "12th":          ["DAV Public School", "Kendriya Vidyalaya", "DPS", "Local Junior College"],
    "Graduate":      ["IIT Delhi", "IIT Bombay", "NIT Trichy", "Delhi University", "Mumbai University",
                      "Anna University", "Pune University", "Osmania University", "Jadavpur University",
                      "Christ University", "Manipal University", "VIT Vellore", "BITS Pilani",
                      "Local Engineering College", "Government Degree College", "Private College"],
    "Post Graduate": ["IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta", "XLRI", "MDI Gurgaon",
                      "IIT Bombay (M.Tech)", "NIT (M.Tech)", "Delhi University (MA)",
                      "AIIMS (MD)", "CMC Vellore", "Local University PG"],
    "PhD":           ["IIT", "IISc Bangalore", "TIFR", "JNU", "Hyderabad University"],
}

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
                 "Hexaware", "Mindtree", "Persistent Systems", "NIIT Technologies"],
    "finance":  ["HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra",
                 "HDFC Life", "LIC", "Deloitte", "PwC India", "KPMG India", "EY India"],
    "govt":     ["Central Government", "State Government", "PSU Company", "DRDO",
                 "ISRO", "Indian Railways", "Indian Army", "Indian Police Service"],
    "medical":  ["Apollo Hospitals", "Fortis Healthcare", "Max Healthcare", "AIIMS",
                 "Private Clinic", "Government Hospital", "Medanta"],
    "education":["IIT", "NIT", "Private University", "Government College", "CBSE School"],
    "business": ["Family Business", "Self Employed", "Partnership Firm", "Proprietorship"],
}

# Suspicious domains used in fraud profiles
SUSPICIOUS_EMAIL_DOMAINS = [
    "mailinator.com", "tempmail.com", "guerrillamail.com", "throwam.com",
    "fakeinbox.com", "yopmail.com", "trashmail.com", "dispostable.com",
]

LEGITIMATE_EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "rediffmail.com",
]

# ── Geography ─────────────────────────────────────────────────────────────────

INDIAN_CITIES = {
    "Tier 1": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad"],
    "Tier 2": ["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Visakhapatnam",
               "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad",
               "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad"],
}
ALL_CITIES = INDIAN_CITIES["Tier 1"] + INDIAN_CITIES["Tier 2"]
CITY_WEIGHTS = [0.04] * 8 + [0.02] * 20  # tier 1 twice as likely
# normalise
_cw = np.array(CITY_WEIGHTS, dtype=float)
CITY_WEIGHTS = (_cw / _cw.sum()).tolist()

INDIAN_STATES = ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "West Bengal",
                 "Telangana", "Gujarat", "Rajasthan", "Uttar Pradesh", "Punjab",
                 "Bihar", "Madhya Pradesh", "Andhra Pradesh", "Kerala", "Haryana"]

COUNTRIES = ["India", "United States", "United Kingdom", "Canada", "Australia",
             "UAE", "Singapore", "Germany", "New Zealand"]
COUNTRY_WEIGHTS = [0.78, 0.07, 0.04, 0.03, 0.03, 0.03, 0.01, 0.005, 0.005]

# ── Income model ──────────────────────────────────────────────────────────────
# income_lpa = base + experience_bonus + field_multiplier + noise
# These are realistic for Indian job market (2024 values in LPA)

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

HOBBIES_POOL = [
    "Reading", "Cooking", "Travelling", "Music", "Movies", "Cricket", "Yoga",
    "Gardening", "Photography", "Badminton", "Trekking", "Painting", "Dancing",
    "Volunteering", "Cycling", "Swimming", "Chess", "Writing", "Meditation",
    "Fitness", "Football", "Tennis", "Singing", "Gaming", "Birdwatching",
]