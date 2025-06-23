import pandas as pd
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import nltk
import numpy as np
from typing import Dict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
import os

if __name__ == "__main__":
    print("hop")

    # Charger le fichier CSV avec le bon séparateur
    df = pd.read_csv("notebooks/Vic/cleaned_data.csv", sep="|", engine="python")

    # Nettoyage basique (optionnel mais recommandé)
    df['modern'] = df['modern'].str.strip()
    df['modern'] = df['modern'].replace("traduction en ancien français :","")
    df['old_french'] = df['old_french'].str.strip().replace("traduction en ancien français :","")

    print(df.head(1))

    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=False)

    class TranslationDataset(Dataset):
        def __init__(self, modern_texts, old_french_texts, tokenizer, max_length=16):
            self.modern_texts = modern_texts
            self.old_french_texts = old_french_texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.modern_texts)
        
        def __getitem__(self, idx):
            modern = self.modern_texts[idx]
            old_french = self.old_french_texts[idx]
            modern_text = f"translate French to OldFrench: {modern}"
            
            # Tokenisation
            inputs = self.tokenizer(
                modern_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            targets = self.tokenizer(
                old_french, 
                max_length=self.max_length, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'labels': targets['input_ids'].flatten()
            }

    # Modèle basé sur mT5 ou mBERT
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(val_df, test_size=0.5, random_state=42)

    # Créer le dossier de sauvegarde s'il n'existe pas
    model_save_path = "./trained_french_oldfrench_model"
    os.makedirs(model_save_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=1,      # Encore plus petit
        per_device_eval_batch_size=1,       # Encore plus petit
        gradient_accumulation_steps=8,      # Simule batch_size=8
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy='steps',              # Changé de 'epoch' à 'steps'
        eval_steps=100,                     # Évalue moins souvent
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        dataloader_pin_memory=False,        # Économise la mémoire
        remove_unused_columns=True,
    )

    train_dataset = TranslationDataset(
        modern_texts=train_df["modern"].tolist(),
        old_french_texts=train_df["old_french"].tolist(),
        tokenizer=tokenizer,
        max_length=64
    )

    val_dataset = TranslationDataset(
        modern_texts=val_df["modern"].tolist(),
        old_french_texts=val_df["old_french"].tolist(),
        tokenizer=tokenizer,
        max_length=64
    )

    def evaluate_model(predictions, references):
        # BLEU Score
        bleu = corpus_bleu(predictions, [references])
        
        # ROUGE Score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        
        # Similarité de caractères (important pour l'ancien français)
        char_similarity = [
            len(set(ref) & set(pred)) / len(set(ref) | set(pred)) if len(set(ref) | set(pred)) > 0 else 0
            for ref, pred in zip(references, predictions)
        ]
        
        return {
            'bleu': bleu.score,
            'rouge1': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
            'char_similarity': np.mean(char_similarity)
        }

    def compute_metrics(eval_pred) -> Dict:
        predictions, labels = eval_pred
        
        # CORRECTION: Convertir les logits en tokens IDs
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Si predictions contient des logits, prendre l'argmax
        if len(predictions.shape) == 3:  # [batch, seq_len, vocab_size]
            predictions = np.argmax(predictions, axis=-1)
        
        # Remplacer les labels -100 par le token pad
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # Décoder les prédictions et labels
        try:
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # Nettoyer les textes vides
            decoded_preds = [pred.strip() if pred.strip() else "vide" for pred in decoded_preds]
            decoded_labels = [label.strip() if label.strip() else "vide" for label in decoded_labels]
            
            return evaluate_model(decoded_preds, decoded_labels)
        
        except Exception as e:
            print(f"Erreur dans compute_metrics: {e}")
            print(f"Shape predictions: {predictions.shape}")
            print(f"Shape labels: {labels.shape}")
            print(f"Type predictions: {type(predictions)}")
            
            # Retourner des métriques par défaut en cas d'erreur
            return {
                'bleu': 0.0,
                'rouge1': 0.0,
                'char_similarity': 0.0
            }
        
    import gc

    # Nettoie la mémoire GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Affiche l'utilisation mémoire
        print(f"Mémoire GPU utilisée: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"Mémoire GPU réservée: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print("GPU non disponible, utilisation du CPU")

    class MemoryCleanupCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"Mémoire nettoyée - Époque {state.epoch}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[MemoryCleanupCallback()]
    )

    print("Début de l'entraînement...")
    trainer.train()
    
    print("Entraînement terminé. Sauvegarde du modèle...")
    
    # Sauvegarder le modèle et le tokenizer
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"Modèle sauvegardé dans: {model_save_path}")
    
    # Optionnel: Sauvegarder aussi les métriques finales
    final_metrics_path = os.path.join(model_save_path, "final_metrics.txt")
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        f.write("Métriques finales de l'entraînement:\n")
        f.write(f"Nombre d'époques: {training_args.num_train_epochs}\n")
        f.write(f"Taille du dataset d'entraînement: {len(train_dataset)}\n")
        f.write(f"Taille du dataset de validation: {len(val_dataset)}\n")
        f.write(f"Learning rate: {training_args.learning_rate}\n")
        f.write(f"Batch size effectif: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}\n")
    
    print(f"Métriques sauvegardées dans: {final_metrics_path}")
    
    # Test rapide du modèle sauvegardé
    print("\nTest du modèle sauvegardé:")
    test_input = "translate French to OldFrench: Salut tout le monde ! Aujourd'hui on a fait un pique-nique dans le jardin. Après j'ai été malade, je me suis vidé par tous les trous..."
    inputs = tokenizer(test_input, return_tensors="pt", max_length=64, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Entrée: {test_input}")
        print(f"Sortie: {translation}")
    
    print("\nScript terminé avec succès !")