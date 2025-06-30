import random

def generate_prompt_modern_text():
    formats = [
        "une lettre personnelle", "une anecdote", "un article de blog", "une petite histoire", "un récit de souvenir",
        "un email", "un message vocal", "un post sur les réseaux sociaux", "un journal intime", "un texte de réflexion"
    ]
    
    primary_topics = [
        "la vie quotidienne", "les études", "le pipi ou le caca", "les amis", "le travail", "le sport", "les loisirs", "la famille",
        "les voyages", "les repas", "la météo", "les émotions", "une rencontre", "la technologie", "le logement",
        "une habitude", "une dispute", "une fête", "un souvenir marquant", "le transport", "une passion", "les problèmes intestinaux",
        "un animal"
    ]
    
    tone_options = [
        "drôle", "sérieux", "émouvant", "neutre", "enthousiaste", "réaliste", "nostalgique", "sincère", "ironique", "burlesque", "vulgaire"
    ]
    
    writing_personas = [
        "un papi de 75 ans qui raconte ses souvenirs",
        "une mamie bienveillante et sage",
        "un jeune de banlieue de 18 ans",
        "une étudiante en art passionnée de 22 ans",
        "un enfant curieux de 12 ans",
        "un cadre parisien stressé de 35 ans",
        "une mère de famille débordée",
        "un retraité qui découvre internet",
        "une ado rebelle de 16 ans",
        "un prof de français passionné",
        "une influenceuse lifestyle de 25 ans",
        "un artisan rural fier de son métier",
        "une grand-mère moderne et branchée",
        "un étudiant en médecine épuisé",
        "une voyageuse solo aventurière",
        "un père célibataire organisé",
        "une lycéenne timide mais observatrice",
        "un chef cuisinier perfectionniste",
        "une bibliothécaire discrète mais cultivée",
        "un sportif motivé et discipliné"
    ]
    
    format_choice = random.choice(formats)
    topic1, topic2 = random.sample(primary_topics, 2)
    tone = random.choice(tone_options)
    persona = random.choice(writing_personas)
    
    prompt = (
        f"Écris {format_choice} {tone} qui parle de {topic1}"
        f"{' et de ' + topic2 if random.random() > 0.4 else ''}. "
        f"Écris ce texte comme si tu étais {persona}. "
        f"Utilise le vocabulaire, les expressions et le style d'écriture typiques de cette personne. "
        f"Fais en sorte que le texte soit personnel et qu'il semble authentique. "
        f"Si tu as besoin d'un nom invente le. "
        f"Le texte doit être en français et faire entre 2 et 4 phrases."
        f"Ne réponds qu'avec le texte, sans introduction ni note."
        f"\n\nTexte :"
    )
    
    return prompt