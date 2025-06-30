import pandas as pd
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
from typing import Dict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainerCallback
import os
import gc
from datetime import datetime

def setup_environment():
    """Configuration optimale pour mT5"""
    import logging
    import warnings
    
    # Réduire les logs verbeux
    logging.getLogger("transformers").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Optimisations PyTorch
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  Attention: Pas de GPU détecté")

class OptimizedTranslationDataset(Dataset):
    """Dataset optimisé pour mT5 avec gestion des langues"""
    
    def __init__(self, modern_texts, old_french_texts, tokenizer, max_length=128):
        self.modern_texts = modern_texts
        self.old_french_texts = old_french_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.modern_texts)
    
    def __getitem__(self, idx):
        modern = str(self.modern_texts[idx]).strip()
        old_french = str(self.old_french_texts[idx]).strip()
        
        # Préfixe optimisé pour mT5 (multilingue)
        input_text = f"translate French to Old French: {modern}"
        
        # Tokenisation avec gestion spéciale pour mT5
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels avec tokenisation cible
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                old_french,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Remplacer les tokens de padding par -100 pour ignorer dans la loss
        labels_input_ids = labels['input_ids'].clone()
        labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': model_inputs['input_ids'].flatten(),
            'attention_mask': model_inputs['attention_mask'].flatten(),
            'labels': labels_input_ids.flatten()
        }

class DetailedMetricsCallback(TrainerCallback):
    """Callback pour métriques détaillées et gestion mémoire"""
    
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.best_score = 0
        
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\n📊 Évaluation - Étape {state.global_step} - Époque {state.epoch:.1f}")
        
        # Nettoyage mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1024**3
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            print(f"🧠 Mémoire GPU: {current_memory:.1f}GB (max: {max_memory:.1f}GB)")
        
        gc.collect()

def enhanced_compute_metrics(tokenizer):
    """Métriques avancées pour évaluer la qualité de la traduction"""
    
    def compute_metrics_fn(eval_pred):
        predictions, labels = eval_pred
        
        # Gérer les différents formats de sortie
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        if len(predictions.shape) == 3:
            predictions = np.argmax(predictions, axis=-1)
        
        # Remplacer -100 par pad_token_id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        try:
            # Décodage avec nettoyage
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Nettoyage post-décodage
            decoded_preds = [pred.strip() for pred in decoded_preds if pred.strip()]
            decoded_labels = [label.strip() for label in decoded_labels if label.strip()]
            
            if not decoded_preds or not decoded_labels:
                return {'bleu': 0.0, 'rouge1': 0.0, 'char_sim': 0.0, 'exact_match': 0.0}
            
            # BLEU score
            bleu = corpus_bleu(decoded_preds, [decoded_labels])
            
            # ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(decoded_labels, decoded_preds)]
            
            # Métriques spécifiques à l'ancien français
            char_similarities = []
            exact_matches = 0
            word_overlaps = []
            
            for ref, pred in zip(decoded_labels, decoded_preds):
                # Similarité de caractères
                ref_chars = set(ref.lower())
                pred_chars = set(pred.lower())
                char_sim = len(ref_chars & pred_chars) / len(ref_chars | pred_chars) if ref_chars | pred_chars else 0
                char_similarities.append(char_sim)
                
                # Correspondance exacte
                if ref.lower().strip() == pred.lower().strip():
                    exact_matches += 1
                
                # Chevauchement de mots
                ref_words = set(ref.lower().split())
                pred_words = set(pred.lower().split())
                word_overlap = len(ref_words & pred_words) / len(ref_words | pred_words) if ref_words | pred_words else 0
                word_overlaps.append(word_overlap)
            
            # Afficher quelques exemples
            if len(decoded_preds) >= 3:
                print("\n🔍 Exemples de traduction:")
                for i in range(min(3, len(decoded_preds))):
                    print(f"  Référence : '{decoded_labels[i]}'")
                    print(f"  Prédiction: '{decoded_preds[i]}'")
                    print()
            
            metrics = {
                'bleu': bleu.score,
                'rouge1': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
                'rouge2': np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
                'rougeL': np.mean([s['rougeL'].fmeasure for s in rouge_scores]),
                'char_sim': np.mean(char_similarities),
                'exact_match': exact_matches / len(decoded_labels) * 100,
                'word_overlap': np.mean(word_overlaps)
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Erreur dans compute_metrics: {e}")
            return {'bleu': 0.0, 'rouge1': 0.0, 'char_sim': 0.0, 'exact_match': 0.0}
    
    return compute_metrics_fn

def main():
    print("🏰 === ENTRAÎNEMENT mT5 pour FRANÇAIS → ANCIEN FRANÇAIS ===")
    print(f"🕐 Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    setup_environment()
    
    # Configuration du modèle
    MODEL_CONFIGS = {
        'mt5-small': {
            'name': 'google/mt5-small',
            'batch_size': 8,
            'grad_accum': 2,
            'lr': 5e-4,
            'epochs': 12
        },
        'mt5-base': {
            'name': 'google/mt5-base', 
            'batch_size': 4,
            'grad_accum': 4,
            'lr': 3e-4,
            'epochs': 10
        },
        'mt5-large': {
            'name': 'google/mt5-large',
            'batch_size': 2,
            'grad_accum': 8,
            'lr': 1e-4,
            'epochs': 8
        }
    }
    
    # Choix du modèle (modifiez ici selon votre machine)
    MODEL_SIZE = 'mt5-base'  # Changez selon vos ressources
    config = MODEL_CONFIGS[MODEL_SIZE]
    
    print(f"🤖 Modèle sélectionné: {config['name']}")
    print(f"📊 Batch effectif: {config['batch_size'] * config['grad_accum']}")
    print(f"🎯 Learning rate: {config['lr']}")
    print(f"🔄 Époques: {config['epochs']}\n")
    
    # Chargement des données
    print("📁 Chargement du dataset...")
    df = pd.read_csv("../data/dataset.csv", sep="\\|\\|\\|", engine="python")
    df = df[['modern', 'old_french']].dropna()
    df['modern'] = df['modern'].str.strip()
    df['old_french'] = df['old_french'].str.strip()
    
    print(f"✅ Dataset: {len(df)} paires de traduction")
    
    # Split des données
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"📊 Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Chargement du modèle mT5
    print(f"\n🔄 Chargement de {config['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['name'])
    model = AutoModelForSeq2SeqLM.from_pretrained(config['name'])
    
    # Configuration des datasets
    max_length = 128
    train_dataset = OptimizedTranslationDataset(
        train_df['modern'].tolist(),
        train_df['old_french'].tolist(),
        tokenizer,
        max_length
    )
    
    val_dataset = OptimizedTranslationDataset(
        val_df['modern'].tolist(),
        val_df['old_french'].tolist(),
        tokenizer,
        max_length
    )
    
    # Dossier de sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./mt5_french_oldfrench_{MODEL_SIZE}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Arguments d'entraînement optimisés pour mT5
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['lr'],
        warmup_steps=500,
        weight_decay=0.01,
        
        # Stratégie d'évaluation et sauvegarde
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='bleu',
        greater_is_better=True,
        
        # Optimisations
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        
        # Logging
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        report_to=None,  # Désactiver wandb/tensorboard
        
        # Optimisations mémoire
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )
    
    # Callback personnalisé
    metrics_callback = DetailedMetricsCallback(val_dataset, tokenizer)
    
    # Création du trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=enhanced_compute_metrics(tokenizer),
        callbacks=[metrics_callback]
    )
    
    # Entraînement
    print(f"\n🚀 Début de l'entraînement...")
    print("=" * 60)
    
    trainer.train()
    
    # Sauvegarde finale
    print(f"\n💾 Sauvegarde du modèle dans {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Informations finales
    info_file = os.path.join(output_dir, "training_summary.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"=== RÉSUMÉ ENTRAÎNEMENT mT5 ===\n")
        f.write(f"Modèle: {config['name']}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {len(df)} exemples\n")
        f.write(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}\n")
        f.write(f"Époques: {config['epochs']}\n")
        f.write(f"Batch effectif: {config['batch_size'] * config['grad_accum']}\n")
        f.write(f"Learning rate: {config['lr']}\n")
        f.write(f"Max length: {max_length}\n")
    
    print(f"✅ Entraînement terminé!")
    print(f"📁 Modèle: {output_dir}")
    print(f"📋 Résumé: {info_file}")
    print(f"🕐 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()