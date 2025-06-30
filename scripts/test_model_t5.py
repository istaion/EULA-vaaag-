import os
import warnings
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

# Supprimer les avertissements non critiques
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_model(model_path):
    """Charge le modèle et le tokenizer depuis le chemin spécifié"""
    print(f"Chargement du modèle depuis : {model_path}")
    
    try:
        # Charger le modèle et le tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        
        # Détecter le device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"Modèle chargé avec succès sur : {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None, None, None

def translate_text(model, tokenizer, device, text):
    """Traduit un texte du français moderne vers l'ancien français"""
    
    # Préparer le texte d'entrée avec le préfixe requis
    input_text = f"translate French to OldFrench: {text}"
    
    # Tokeniser
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=64, 
        truncation=True,
        padding=True
    )
    
    # Déplacer sur le bon device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Générer la traduction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True,
            do_sample=False,
            temperature=1.0
        )
    
    # Décoder la sortie
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation

def main():
    print("=== Test du modèle Français -> Ancien Français ===\n")
    
    # Chemin vers le modèle sauvegardé
    model_path = "./trained_french_oldfrench_model"
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle n'existe pas dans {model_path}")
        print("Assurez-vous d'avoir entraîné le modèle d'abord.")
        return
    
    # Charger le modèle
    model, tokenizer, device = load_model(model_path)
    
    if model is None:
        print("Impossible de charger le modèle. Arrêt du programme.")
        return
    
    # Textes de test
    test_texts = [
        "Bonjour, comment allez-vous ?",
        "Je vous souhaite une bonne journée.",
        "Où habitez-vous ?",
        "Merci beaucoup pour votre aide.",
        "Il fait beau aujourd'hui.",
        "Pouvez-vous m'aider ?",
        "Au revoir et à bientôt."
    ]
    
    print("Traductions :")
    print("-" * 80)
    
    for i, text in enumerate(test_texts, 1):
        try:
            translation = translate_text(model, tokenizer, device, text)
            print(f"{i}. Français moderne : {text}")
            print(f"   Ancien français : {translation}")
            print()
            
        except Exception as e:
            print(f"Erreur lors de la traduction de '{text}' : {e}")
            print()
    
    # Test interactif
    print("\n" + "="*80)
    print("Mode interactif (tapez 'quit' pour quitter) :")
    print("="*80)
    
    while True:
        user_input = input("\nEntrez un texte en français moderne : ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Au revoir !")
            break
            
        if not user_input:
            continue
            
        try:
            translation = translate_text(model, tokenizer, device, user_input)
            print(f"Traduction : {translation}")
            
        except Exception as e:
            print(f"Erreur lors de la traduction : {e}")

if __name__ == "__main__":
    main()