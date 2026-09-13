"""
bio_templates.py

30 hand-crafted seed bios written to sound authentic on Indian matrimonial
platforms. Fraud profiles pick from this pool and apply word-substitution
to make cosine similarity detection non-trivial (detection threshold ~0.75
rather than 1.0, which is what you'd get with exact copies).

Each bio is a dict with:
  - text:     the raw template (use {gender_pronoun}, {city}, {profession} as slots)
  - category: general / career_focused / family_focused / spiritual
  - gender:   Male / Female / Any
"""

BIO_TEMPLATES = [
    {
        "text": (
            "I am a simple and grounded person who values family above everything else. "
            "Having been raised in a close-knit family in {city}, I believe in maintaining "
            "strong bonds with parents and siblings even after marriage. My friends describe "
            "me as someone who is caring, responsible, and easy to talk to. In my free time "
            "I enjoy cooking, reading, and spending quality time with loved ones. I am looking "
            "for a life partner who shares similar family values and is ready to build a "
            "beautiful life together with love and understanding."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "A hardworking professional settled in {city}, I have always believed that a "
            "successful career and a happy family go hand in hand. I work as a {profession} "
            "and take great pride in my work. Outside of office hours, I love travelling to "
            "new places and exploring different cuisines. I come from a traditional yet "
            "open-minded family that respects individual choices. Looking for a partner who "
            "is educated, ambitious, and family-oriented. Caste no bar, but shared values "
            "are important to me."
        ),
        "category": "career_focused",
        "gender": "Any",
    },
    {
        "text": (
            "Born and brought up in {city}, I am the kind of person who finds joy in small "
            "things — a good book, a hot cup of chai, a long drive on a rainy day. I am "
            "close to my parents and believe in respecting elders. My upbringing has taught "
            "me the importance of honesty and commitment. I am a {profession} by profession "
            "and am financially stable and settled. Seeking a life partner who is understanding, "
            "warm-hearted, and ready for a beautiful journey together."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "I consider myself a modern yet traditional person. I respect my culture and "
            "family values but also believe in equality and mutual respect in relationships. "
            "Currently working as a {profession} in {city}, I enjoy a stable and fulfilling "
            "professional life. I love music, trekking, and volunteering for social causes "
            "on weekends. Looking for someone who is kind, mature, and has a positive outlook "
            "on life. Ready to relocate if required."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "Faith and family are the two pillars of my life. I am deeply rooted in my "
            "religious values while also being highly educated and professionally accomplished. "
            "As a {profession} based in {city}, I have achieved a comfortable life through "
            "hard work and dedication. I am looking for a partner who is God-fearing, "
            "educated, and family-oriented. My family is very supportive and will warmly "
            "welcome my life partner. Serious inquiries only."
        ),
        "category": "spiritual",
        "gender": "Any",
    },
    {
        "text": (
            "I am a fun-loving, outgoing person who also knows how to be serious when needed. "
            "Originally from a small town but now settled in {city}, I have worked hard to "
            "build my career as a {profession}. I believe in honest communication and think "
            "that trust is the foundation of any relationship. My hobbies include photography, "
            "playing cricket, and watching documentaries. Looking for a partner who is "
            "independent, confident, and ready to grow together as a team."
        ),
        "category": "general",
        "gender": "Male",
    },
    {
        "text": (
            "A warm, independent woman who knows her mind. I am a {profession} working in "
            "{city} and proud to be self-sufficient. My family has always encouraged me to "
            "pursue my dreams while staying rooted in our values. I enjoy cooking, yoga, "
            "and reading fiction. I am looking for an understanding and respectful partner "
            "who values a woman's career and independence. My family is open-minded and "
            "will support us in building our life together."
        ),
        "category": "career_focused",
        "gender": "Female",
    },
    {
        "text": (
            "Life for me is about balance — career, family, health, and happiness. Currently "
            "a {profession} in {city}, I work hard during the week and like to unwind by "
            "cooking new recipes or going for a run on weekends. I come from a middle-class "
            "family with strong values. My parents are retired and I am responsible for them. "
            "Looking for a partner who understands the importance of family responsibilities "
            "and is emotionally mature."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "I believe marriage is a beautiful partnership built on trust, love, and mutual "
            "respect. As someone who has seen strong and healthy relationships in my own "
            "family, I know what it takes to make a marriage work. I am a {profession} "
            "based out of {city} and lead a fairly settled life. In my free time I enjoy "
            "travelling, listening to music, and spending time with my nieces and nephews. "
            "Looking for a kind-hearted and sincere partner."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "Simple, educated, and values-driven. I was raised in a joint family in {city} "
            "and have always seen firsthand the importance of compromise and understanding "
            "in relationships. I work as a {profession} and have a stable income and "
            "good savings. I enjoy gardening, badminton, and watching historical documentaries. "
            "Looking for an educated life partner from a good family background. "
            "Willing to consider proposals from all communities."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "I describe myself as someone who is ambitious in professional life but grounded "
            "and humble at home. Born in a small town, I moved to {city} for education and "
            "stayed on after landing a good position as a {profession}. My journey has taught "
            "me the value of perseverance and gratitude. I am close to my parents and siblings "
            "and visit them every holiday. Looking for a sincere partner with good values "
            "who wants to build a home full of love and laughter."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "If I were to describe myself in one word it would be: genuine. What you see "
            "is what you get. I am a straightforward {profession} in {city} who has no "
            "time for games or dishonesty. My family means the world to me and I make sure "
            "to call my parents every single day. On weekends I like to go trekking, play "
            "chess, or simply relax at home with a good movie. Seeking a partner who is "
            "equally genuine, kind, and ready for a serious commitment."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "Career-driven but not at the cost of family. I have worked very hard to reach "
            "where I am today as a {profession} in {city}, but I never lost sight of what "
            "truly matters — relationships, health, and inner peace. I practice yoga every "
            "morning and believe in a healthy lifestyle. I come from a well-educated family "
            "with progressive thinking. Looking for a life partner who is my equal — "
            "professionally successful, emotionally intelligent, and family-oriented."
        ),
        "category": "career_focused",
        "gender": "Female",
    },
    {
        "text": (
            "I am a responsible and caring individual looking for a life partner to share "
            "life's journey with. Currently working as a {profession} and residing in {city}, "
            "I live with my parents and younger sibling. Our family is very close and we "
            "celebrate every festival together. I enjoy cooking, painting, and long walks "
            "in the evening. I am looking for someone who is patient, understanding, and "
            "has a loving heart. Compatibility and chemistry matter more to me than status."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "Adventure is my middle name — but so is responsibility. I love trekking, "
            "travelling, and trying street food from different cities. Professionally I am "
            "a {profession} working in {city}, and I take my work very seriously. When it "
            "comes to relationships I believe in open communication and giving each other "
            "space while still being each other's strongest support. Looking for someone "
            "who is fun, grounded, and genuinely good at heart."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "My life philosophy: work hard, stay humble, love deeply. I am a {profession} "
            "settled in {city} after studying in one of the premier institutions in India. "
            "My parents have been my biggest inspiration and I hope to find a partner who "
            "respects that bond. I love reading non-fiction, cycling, and cooking for my "
            "family on Sunday mornings. Looking for a well-educated, ambitious, and "
            "emotionally stable partner. Ready to meet families soon after initial connection."
        ),
        "category": "career_focused",
        "gender": "Any",
    },
    {
        "text": (
            "A God-fearing, educated professional from {city}. I believe that a marriage "
            "blessed by elders and built on faith and honesty will stand the test of time. "
            "I work as a {profession} and am financially independent and responsible. "
            "My hobbies include singing devotional songs, reading scriptures, and social work. "
            "Looking for a life partner from a religious, well-educated family. "
            "We are open to alliance from our community and closely related communities."
        ),
        "category": "spiritual",
        "gender": "Any",
    },
    {
        "text": (
            "I grew up watching my parents set a beautiful example of what a marriage should "
            "be — full of respect, laughter, and unwavering support. That is exactly what I "
            "want to recreate. I am a {profession} in {city}, financially settled, and "
            "emotionally ready for marriage. I enjoy movies, music, cooking, and cycling. "
            "I have a sharp sense of humor and believe life is too short to be serious "
            "all the time. Looking for someone who can laugh with me and grow with me."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "I come from a large and loving family where every occasion is celebrated with "
            "great enthusiasm. Festivals, birthdays, anniversaries — we never miss a chance "
            "to come together. As a {profession} in {city}, I have built a stable life but "
            "I know that true happiness lies in having the right person to share it with. "
            "Looking for a warm, family-oriented partner who values togetherness. "
            "Hobbies include cooking, gardening, and playing carrom with family."
        ),
        "category": "family_focused",
        "gender": "Female",
    },
    {
        "text": (
            "Professionally accomplished, personally grounded. I have had the privilege of "
            "studying and working in some of the best institutions in India and am now "
            "settled as a {profession} in {city}. Despite my career, I remain very close "
            "to my roots and visit my hometown regularly. I believe a life partner should "
            "be your best friend first. Looking for someone who is intelligent, kind, "
            "and secure enough to support and celebrate each other's success."
        ),
        "category": "career_focused",
        "gender": "Any",
    },
    {
        "text": (
            "Honest, dependable, and ready to start the next beautiful chapter of life. "
            "I am a {profession} living in {city} with a good job and a happy family. "
            "I have always believed in long-term commitment and am not interested in "
            "casual connections. My family has given me the freedom to choose my partner "
            "but they will also be involved in the final decision as is our tradition. "
            "Seeking a well-educated, grounded, and affectionate life partner."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "I am an introvert who becomes an extrovert once comfortable — my friends and "
            "colleagues will testify to that! Working as a {profession} in {city}, I lead "
            "a structured and balanced life. I meditate every morning and believe strongly "
            "in mental and emotional health. I love reading, writing, and watching indie "
            "cinema. Looking for a partner who is thoughtful, empathetic, and values "
            "deep conversations over surface-level compatibility."
        ),
        "category": "spiritual",
        "gender": "Any",
    },
    {
        "text": (
            "Family first — always. As the eldest child in my family, I have always taken "
            "on responsibilities with pride. My parents have sacrificed a lot for my "
            "education and I am now a {profession} in {city} with a stable income. "
            "I take care of my family and will continue to do so after marriage as well. "
            "Looking for a partner who understands and respects family responsibilities. "
            "Our home will always be a welcoming place for both families."
        ),
        "category": "family_focused",
        "gender": "Male",
    },
    {
        "text": (
            "I believe actions speak louder than words. Rather than describing myself at "
            "length, I will let my life speak for itself — a {profession} in {city}, "
            "a responsible son/daughter, a loyal friend, and hopefully soon, a loving "
            "life partner. I am not looking for perfection — just someone who is real, "
            "kind, and genuinely interested in building a life together. My hobbies "
            "include fitness, travelling, and street photography."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "A small-town heart in a big-city life. I moved to {city} to pursue my dreams "
            "and today I work as a proud {profession}. But at the end of the day, I still "
            "love simple joys — home-cooked food, evening walks, and good conversations. "
            "My parents mean everything to me and I speak to them twice a day. "
            "Looking for a partner who is genuine, caring, and ready to build a beautiful "
            "simple life together away from pretense and show-off."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "I am a strong believer in the institution of marriage and the joy it brings "
            "to two families. I was brought up with good values in {city} and currently "
            "work as a {profession}. My parents and I are very close and they have been "
            "patiently looking for the right match for me. Looking for a homely, educated, "
            "and well-mannered partner. Preferably from our community but open to "
            "considering other alliances if values match."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "Life is what you make of it — and I intend to make mine beautiful. Currently "
            "working as a {profession} in {city}, I am financially secure and emotionally "
            "mature. I enjoy cooking elaborate meals on weekends, practicing yoga, and "
            "watching cricket matches with my family. I come from a very united family and "
            "we are all very involved in each other's lives. Looking for a partner who "
            "is warm, joyful, and ready to embrace family as her/his own."
        ),
        "category": "family_focused",
        "gender": "Any",
    },
    {
        "text": (
            "I am someone who has always followed the principle of working hard and "
            "trusting the process. Today as a {profession} in {city} I feel I have "
            "reached a good stage in life to settle down. My family is supportive and "
            "very welcoming. I love music — both classical and contemporary — and play "
            "the guitar as a hobby. Looking for a life partner who is independent, "
            "has her/his own identity, and yet values the beauty of togetherness."
        ),
        "category": "general",
        "gender": "Any",
    },
    {
        "text": (
            "Rooted in tradition, open to the modern world. I respect the values my "
            "parents taught me while also believing in gender equality and personal "
            "freedom within a relationship. As a {profession} in {city}, I enjoy a "
            "comfortable and purposeful life. I volunteer with an NGO on weekends "
            "and believe in giving back to society. Looking for a partner who is "
            "compassionate, educated, and has a sense of purpose beyond personal success."
        ),
        "category": "spiritual",
        "gender": "Any",
    },
    {
        "text": (
            "I will be honest — I am terrible at writing about myself! But I will try. "
            "I am a {profession} in {city} who loves his/her work and loves coming home "
            "to family even more. I am the kind of person who remembers everyone's "
            "birthdays, cooks something special on anniversaries, and cries at emotional "
            "movies. Looking for someone who is genuine, a little silly, a lot kind, "
            "and ready for a lifetime of beautiful ordinary days together."
        ),
        "category": "general",
        "gender": "Any",
    },
]

# ── Partner preference templates ──────────────────────────────────────────────
# These are appended to bios to make profiles feel complete

PARTNER_PREF_TEMPLATES = [
    "Looking for an educated, well-settled partner between {age_min} and {age_max} years of age. "
    "Should be family-oriented and respectful. Caste no bar for the right person.",

    "Seeking a partner who is professionally accomplished and emotionally mature. "
    "Age {age_min}-{age_max}. Should be open to staying in {city} after marriage.",

    "Looking for a kind and understanding life partner. Age no bar if compatibility is there. "
    "The person should be close to their family and have good values.",

    "Seeking a well-educated partner from a good family background. "
    "Height preference: above {height_pref}cm. Ready to meet at the earliest.",

    "Looking for a soulmate who values honesty and family above all. "
    "Should be earning well and financially responsible. Age {age_min}-{age_max}.",
]