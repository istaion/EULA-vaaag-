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

def translate_with_mbart(model, tokenizer, device, text, max_length=64):
    """Traduit avec mBART en utilisant différentes stratégies"""
    
    # Configuration langue
    tokenizer.src_lang = "fr_XX"
    
    strategies = {
        "Greedy": {
            "num_beams": 1,
            "do_sample": False
        },
        "Beam Search": {
            "num_beams": 4,
            "do_sample": False,
            "early_stopping": True
        },
        "Sampling": {
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9
        },
        "Creative": {
            "num_beams": 2,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        }
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
        
        # Tester chaque stratégie
        for strategy_name, params in strategies.items():
            try:
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"],
                        max_length=max_length,
                        pad_token_id=tokenizer.pad_token_id,
                        **params
                    )
                
                end_time = time.time()
                
                # Décoder
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results[strategy_name] = {
                    'text': result,
                    'time': end_time - start_time
                }
                
            except Exception as e:
                results[strategy_name] = {
                    'text': f"❌ Erreur: {e}",
                    'time': 0
                }
    
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return {}
    
    return results

def main():
    # Chemin du modèle
    model_path = "./mbart_fast_1339"
    
    # Charger le modèle
    model, tokenizer, device = load_mbart_model(model_path)
    
    if model is None:
        return
    
    # Phrases de test
    test_sentences = [
        "Bonjour, comment allez-vous ?",
        "Je suis très heureux de vous voir aujourd'hui.",
        "Cette belle maison appartient à notre famille.",
        "Nous irons au marché demain matin.",
        "Il fait un temps magnifique en cette saison.",
        "Pouvez-vous m'aider à porter cette lourde charge ?",
        "Mon père travaille dans les champs chaque jour.",
        "Les enfants jouent dans la cour du château.",
        "Nous avons mangé un excellent repas hier soir.",
        "La reine porte une robe de soie brodée d'or.",
        "Les chevaliers partent en guerre contre l'ennemi.",
        "Dame Marguerite prie dans la chapelle du monastère.",
        "Le roi convoque ses vassaux pour le conseil.",
        "Les marchands vendent leurs produits sur la place."
    ]
    
    print(f"\n🧪 TEST AVEC {len(test_sentences)} PHRASES")
    print("=" * 80)
    
    # Tester chaque phrase avec toutes les stratégies
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i:2d}. PHRASE: {sentence}")
        print("-" * 60)
        
        results = translate_with_mbart(model, tokenizer, device, sentence)
        
        if results:
            for strategy, result in results.items():
                time_str = f"({result['time']:.2f}s)" if result['time'] > 0 else ""
                print(f"   {strategy:12s}: {result['text']} {time_str}")
        else:
            print("   ❌ Impossible de générer des traductions")
    
    # Test de performance
    print(f"\n⚡ TEST DE PERFORMANCE")
    print("=" * 30)
    
    perf_sentence = "Bonjour, comment vous portez-vous ce jour ?"
    print(f"Phrase test: {perf_sentence}")
    
    # Test 5 fois pour avoir une moyenne
    times = []
    for i in range(5):
        results = translate_with_mbart(model, tokenizer, device, perf_sentence)
        if results and 'Greedy' in results:
            times.append(results['Greedy']['time'])
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"Temps moyen de génération: {avg_time:.3f}s")
        print(f"Vitesse: {1/avg_time:.1f} traductions/seconde")
    
    # Mode interactif
    print(f"\n🎮 === MODE INTERACTIF ===")
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    print("=" * 40)
    
    while True:
        user_input = input("\nFrançais moderne ➤ ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir !")
            break
            
        if user_input.lower() == 'help':
            print("\n📖 AIDE:")
            print("- Entrez une phrase en français moderne")
            print("- Le modèle la traduira vers l'ancien français")
            print("- Plusieurs stratégies sont testées")
            print("- 'quit' pour quitter")
            continue
            
        if not user_input:
            continue
        
        print(f"\n🔄 Traduction en cours...")
        results = translate_with_mbart(model, tokenizer, device, user_input)
        
        if results:
            print(f"\n📝 Résultats:")
            for strategy, result in results.items():
                time_str = f"({result['time']:.2f}s)" if result['time'] > 0 else ""
                print(f"  {strategy:12s}: {result['text']} {time_str}")
        else:
            print("❌ Impossible de générer une traduction")

def compare_with_original():
    """Compare le modèle entraîné avec mBART original"""
    
    print(f"\n🔍 === COMPARAISON AVEC mBART ORIGINAL ===")
    
    # Charger modèle original
    try:
        print("🔄 Chargement mBART original...")
        original_tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50')
        original_model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        original_model = original_model.to(device)
        
        print("✅ mBART original chargé")
        
    except Exception as e:
        print(f"❌ Impossible de charger mBART original: {e}")
        return
    
    # Charger modèle entraîné
    trained_model, trained_tokenizer, device = load_mbart_model("./mbart_fast_1339")
    
    if trained_model is None:
        return
    
    # Phrases de test
    test_sentences = [
        "Bonjour, comment allez-vous ?",
        "Je suis heureux de vous voir.",
        "Cette maison est belle."
    ]
    
    print(f"\n📊 COMPARAISON:")
    print("=" * 60)
    
    for sentence in test_sentences:
        print(f"\nPhrase: {sentence}")
        print("-" * 40)
        
        # Modèle original
        try:
            original_tokenizer.src_lang = "fr_XX"
            inputs = original_tokenizer(sentence, return_tensors="pt", max_length=64, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = original_model.generate(
                    **inputs,
                    forced_bos_token_id=original_tokenizer.lang_code_to_id["fr_XX"],
                    max_length=64,
                    num_beams=2
                )
            
            original_result = original_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Original : {original_result}")
            
        except Exception as e:
            print(f"Original : ❌ Erreur - {e}")
        
        # Modèle entraîné
        trained_results = translate_with_mbart(trained_model, trained_tokenizer, device, sentence)
        if trained_results and 'Beam Search' in trained_results:
            print(f"Entraîné: {trained_results['Beam Search']['text']}")
        else:
            print(f"Entraîné: ❌ Erreur")

if __name__ == "__main__":
    main()
    
    # Proposer comparaison
    response = input(f"\n🔍 Voulez-vous comparer avec mBART original ? (y/n): ").strip().lower()
    if response in ['y', 'yes', 'oui', 'o']:
        compare_with_original()