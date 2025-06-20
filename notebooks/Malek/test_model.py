import pandas as pd

# Charger le fichier CSV avec le bon séparateur
df = pd.read_csv("../../data/dataset.csv", sep="\\|\\|\\|", engine="python")

# Ne conserver que les colonnes utiles
df = df[['modern', 'old_french']].dropna()

# Nettoyage basique (optionnel mais recommandé)
df['modern'] = df['modern'].str.strip()
df['old_french'] = df['old_french'].str.strip()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=False)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

class TranslationDataset(Dataset):
    def __init__(self, modern_texts, old_french_texts, tokenizer, max_length=128):
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
            # 'input_ids': inputs['input_ids'].flatten(),
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'decoder_attention_mask': targets['attention_mask'].squeeze(),
            'labels': targets['input_ids'].flatten()
        }

# Modèle basé sur mT5 ou mBERT
model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, val_df = train_test_split(val_df, test_size=0.5, random_state=42)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    fp16=True,
    dataloader_pin_memory=False,
)

train_dataset = TranslationDataset(
    modern_texts=train_df["modern"].tolist(),
    old_french_texts=train_df["old_french"].tolist(),
    tokenizer=tokenizer,
    max_length=24
)

val_dataset = TranslationDataset(
    modern_texts=val_df["modern"].tolist(),
    old_french_texts=val_df["old_french"].tolist(),
    tokenizer=tokenizer,
    max_length=24
)



from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
import nltk
import numpy as np
from typing import Dict
import torch

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



import torch

torch.cuda.empty_cache()           # Libère la mémoire inutilisée (mais allouée)
torch.cuda.ipc_collect()           # Nettoie les handles CUDA obsolètes (utile en Notebook)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Exemple d'entrée
modern_text = "Salut tout le monde ! Aujourd'hui on a fait un pique-nique dans le jardin. Après j'ai été malade, je me suis vidé par tous les trous..."
input_text = f"translate French to OldFrench: {modern_text}"

# Tokenisation
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Génération (inférence)
with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

# Décodage
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("📝 Texte d'origine :", modern_text)
print("🏰 Traduction en vieux français :", output_text)





