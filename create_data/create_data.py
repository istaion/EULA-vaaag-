import os
import requests
import csv
import time
from dotenv import load_dotenv
from prompt import generate_prompt_modern_text
from openai import OpenAI

# Charger les clés API depuis .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
DATASET_CSV_PATH = "dataset.csv"

def generate_prompt_old_french_translation(modern_text):
    return (
        f"Traduis uniquement ce texte en ancien français du XIe siècle, sans ajout ni explication. "
        f"Ne réponds qu'avec le texte traduit, sans introduction.\n\nTexte à traduire :\n\"{modern_text}\"\n\nTraduction :"
    )

def call_groq_chat(prompt, model="gemma2-9b-it", temperature=0.8, max_tokens=400, max_retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            if not response.ok:
                print(f"[❌ Erreur API Groq - Tentative {attempt + 1}/{max_retries}]")
                print("Status code :", response.status_code)
                print("Message :", response.text)
                
                # Si c'est une erreur 503 et qu'il reste des tentatives
                if response.status_code == 503 and attempt < max_retries - 1:
                    print("⏳ Attente de 1 seconde avant nouvelle tentative...")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                print(f"⏳ Erreur HTTP, nouvelle tentative dans 1 seconde...")
                time.sleep(1)
                continue
            else:
                print(f"❌ Échec après {max_retries} tentatives")
                raise e
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⏳ Erreur inattendue, nouvelle tentative dans 1 seconde...")
                time.sleep(1)
                continue
            else:
                print(f"❌ Échec après {max_retries} tentatives")
                raise e

def call_gpt_chat(prompt, model="gpt-4o", temperature=0.8, max_tokens=400):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def generate_dataset_entry(model_provider="openai", model_name="gpt-4o"):
    modern_prompt = generate_prompt_modern_text()
    print(f"[Prompt moderne]\n{modern_prompt}\n")
    
    modern_text = call_groq_chat(modern_prompt, model="gemma2-9b-it")
    print(f"[Texte moderne généré]\n{modern_text}\n")
    
    translation_prompt = generate_prompt_old_french_translation(modern_text)
    print(f"[Prompt traduction]\n{translation_prompt}\n")
    
    if model_provider == "groq":
        old_french_text = call_groq_chat(translation_prompt, model="gemma2-9b-it")
    elif model_provider == "openai":
        old_french_text = call_gpt_chat(translation_prompt, model=model_name)
    
    print(f"[Texte ancien généré]\n{old_french_text}\n")
    
    return {
        "modern": modern_text,
        "old_french": old_french_text,
        "modern_prompt": modern_prompt,
        "translation_prompt": translation_prompt
    }

def clean_text_for_csv(text):
    """Supprime les retours à la ligne pour ne pas casser le CSV."""
    return text.replace("\n", " ").replace("\r", " ").strip()

def save_to_custom_csv(entry, file_path):
    file_exists = os.path.isfile(file_path)
    write_header = not file_exists or os.stat(file_path).st_size == 0
    
    with open(file_path, "a", encoding="utf-8", newline='') as csvfile:
        if write_header:
            csvfile.write("modern|||old_french|||modern_prompt|||translation_prompt\n")
        
        row = "|||".join([
            clean_text_for_csv(entry["modern"]),
            clean_text_for_csv(entry["old_french"]),
            clean_text_for_csv(entry["modern_prompt"]),
            clean_text_for_csv(entry["translation_prompt"])
        ])
        csvfile.write(row + "\n")

def main():
    target_size = int(input("Combien d'exemples veux-tu générer ? "))
    
    for i in range(target_size):
        print(f"\n--- Exemple {i+1}/{target_size} ---")
        try:
            entry = generate_dataset_entry()
            save_to_custom_csv(entry, DATASET_CSV_PATH)
        except Exception as e:
            print(f"❌ Erreur lors de la génération de l'exemple {i+1}: {e}")
            print("🔄 Passage à l'exemple suivant...")
            continue
    
    print(f"\n✅ Dataset enrichi avec {target_size} nouveaux exemples dans {DATASET_CSV_PATH}.")

if __name__ == "__main__":
    main()