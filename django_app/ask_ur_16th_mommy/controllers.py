import os
import requests
import csv
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_prompt_old_french_translation(modern_text):
    return (
        f"Traduis uniquement ce texte en ancien français, sans ajout ni explication."
        f"Le résultat doit être amusant."
        f"Ne réponds qu'avec le texte traduit, sans introduction.\n\nTexte à traduire :\n\"{modern_text}\"\n\nTraduction :"
    )

def call_groq_chat(prompt, model="gemma2-9b-it", temperature=0.8, max_tokens=400, max_retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    prompt = generate_prompt_old_french_translation(prompt)
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