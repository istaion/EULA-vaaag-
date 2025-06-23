import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
from datetime import datetime
import time

def load_mbart_model(model_path):
    """Charge le modèle mBART entraîné"""
    
    print(f"🌍 === TEST MODÈLE mBART ===")
    print(f"📁 Modèle: {model_path}")
    print(f"🕐 {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return None, None, None
    
    try:
        print("🔄 Chargement du modèle mBART...")
        
        # Charger tokenizer et modèle
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ Modèle chargé sur: {device}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 VRAM utilisée: {allocated:.1f}GB")
        
        # Vérifier les langues disponibles
        print(f"\n🔍 Informations mBART:")
        print(f"  Vocabulaire: {len(tokenizer)}")
        print(f"  Token français: fr_XX = {tokenizer.lang_code_to_id.get('fr_XX', 'Non trouvé')}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None, None

def translate_with_mbart(model_path, text, max_length=1024):
    """Traduit avec mBART en utilisant différentes stratégies"""
    
    model, tokenizer, device = load_mbart_model(model_path)

    # Configuration langue
    tokenizer.src_lang = "fr_XX"
    
    strategie = {
            "num_beams": 2,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
    
    results = {}
    
    try:
        # Tokeniser l'entrée
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
        try:
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"],
                    max_length=max_length,
                    pad_token_id=tokenizer.pad_token_id,
                    **strategie
                )
            
            end_time = time.time()
            
            # Décoder
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results = {
                'text': result,
                'time': end_time - start_time
            }
            
        except Exception as e:
            results = {
                'text': f"❌ Erreur: {e}",
                'time': 0
            }
    
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return {}
    
    return results
