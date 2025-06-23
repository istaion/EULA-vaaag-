import pandas as pd
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import gc
from datetime import datetime

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class FastMBartDataset(Dataset):
    """Dataset ultra-rapide pour mBART"""
    
    def __init__(self, modern_texts, old_french_texts, tokenizer, max_length=64):
        self.modern_texts = modern_texts
        self.old_french_texts = old_french_texts
        self.tokenizer = tokenizer
        self.max_length = max_length  # RÉDUIT à 64
        
    def __len__(self):
        return len(self.modern_texts)
    
    def __getitem__(self, idx):
        modern = str(self.modern_texts[idx]).strip()
        old_french = str(self.old_french_texts[idx]).strip()
        
        # Configuration langue
        self.tokenizer.src_lang = "fr_XX"
        self.tokenizer.tgt_lang = "fr_XX"
        
        # Tokenisation RAPIDE
        inputs = self.tokenizer(
            modern,
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
        
        labels = targets['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def test_speed(model, tokenizer, device):
    """Test de vitesse"""
    
    print("\n⚡ Test de vitesse:")
    
    tokenizer.src_lang = "fr_XX"
    sentence = "Bonjour, comment allez-vous ?"
    
    import time
    start = time.time()
    
    inputs = tokenizer(sentence, return_tensors="pt", max_length=64, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"],
            max_length=64,
            num_beams=2,  # Moins de beams = plus rapide
            early_stopping=True
        )
    
    end = time.time()
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"  Temps: {end - start:.2f}s")
    print(f"  Résultat: {result}")

def main():
    print("⚡ === mBART ULTRA-RAPIDE ===")
    print(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    if not torch.cuda.is_available():
        print("❌ GPU requis!")
        return
    
    print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    memory_cleanup()
    
    # Configuration ULTRA-RAPIDE
    SPEED_CONFIG = {
        'model_name': 'facebook/mbart-large-50',  # Plus petit que many-to-many
        'batch_size': 8,                          # BEAUCOUP plus grand
        'grad_accum': 1,                          # Pas d'accumulation
        'max_length': 64,                         # TRÈS réduit
        'lr': 5e-5,                              # LR plus élevé
        'epochs': 3                              # Moins d'époques
    }
    
    print(f"🤖 Modèle: {SPEED_CONFIG['model_name']}")
    print(f"📦 Batch size: {SPEED_CONFIG['batch_size']}")
    print(f"📏 Max length: {SPEED_CONFIG['max_length']}")
    print(f"⚡ Config: VITESSE MAXIMALE")
    
    # Dataset RÉDUIT pour test rapide
    print("\n📁 Chargement dataset...")
    df = pd.read_csv("../notebooks/Vic/cleaned_data.csv", sep="|", engine="python")
    df = df[['modern', 'old_french']].dropna()
    df['modern'] = df['modern'].str.strip()
    df['old_french'] = df['old_french'].str.strip()
    
    # TRÈS réduit pour test de vitesse
    df = df.sample(n=min(800, len(df)), random_state=42)
    print(f"📊 Dataset RÉDUIT: {len(df)} exemples pour test vitesse")
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Modèle
    print(f"\n🔄 Chargement {SPEED_CONFIG['model_name']}...")
    tokenizer = MBart50TokenizerFast.from_pretrained(SPEED_CONFIG['model_name'])
    model = MBartForConditionalGeneration.from_pretrained(SPEED_CONFIG['model_name'])
    
    device = torch.device('cuda')
    model = model.to(device)
    
    print("✅ mBART chargé")
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 VRAM: {allocated:.1f}GB")
    
    # Test de vitesse
    test_speed(model, tokenizer, device)
    
    # Dataset rapide
    train_dataset = FastMBartDataset(
        train_df['modern'].tolist(),
        train_df['old_french'].tolist(),
        tokenizer,
        SPEED_CONFIG['max_length']
    )
    
    # Output
    timestamp = datetime.now().strftime("%H%M")
    output_dir = f"./mbart_fast_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Arguments OPTIMISÉS VITESSE
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=SPEED_CONFIG['epochs'],
        per_device_train_batch_size=SPEED_CONFIG['batch_size'],  # GROS batch
        gradient_accumulation_steps=SPEED_CONFIG['grad_accum'],   # Pas d'accumulation
        learning_rate=SPEED_CONFIG['lr'],
        warmup_steps=50,                          # Très peu de warmup
        weight_decay=0.01,
        
        # AUCUNE évaluation = vitesse MAX
        eval_strategy='no',
        save_strategy='epoch',
        save_total_limit=1,                       # Un seul save
        
        # Optimisations VITESSE
        fp16=True,                                # ESSENTIEL
        dataloader_num_workers=4,                 # Plus de workers
        remove_unused_columns=True,
        dataloader_pin_memory=True,               # Plus rapide
        gradient_checkpointing=False,             # DÉSACTIVÉ pour vitesse
        max_grad_norm=1.0,
        
        # Logging minimal
        logging_steps=10,                         # Très fréquent pour monitoring
        report_to=None,
        dataloader_drop_last=True,
        
        # Optimiseur rapide
        optim="adamw_torch",
        lr_scheduler_type="constant",             # Pas de scheduler
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print(f"\n⚡ ENTRAÎNEMENT RAPIDE...")
    print(f"🎯 Objectif: <5 secondes par step")
    print("=" * 40)
    
    try:
        memory_cleanup()
        
        # Chronométrage
        import time
        start_time = time.time()
        
        # Entraînement
        result = trainer.train()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n⏱️  TEMPS TOTAL: {total_time/60:.1f} minutes")
        print(f"⚡ Vitesse: {total_time/len(train_dataset)*SPEED_CONFIG['batch_size']:.2f}s par exemple")
        
        # Test rapide
        print(f"\n🧪 Test post-entraînement:")
        test_speed(model, tokenizer, device)
        
        print(f"\n💾 Sauvegarde...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        if hasattr(result, 'training_loss'):
            print(f"📈 Loss finale: {result.training_loss:.4f}")
        
        print(f"✅ MODÈLE RAPIDE sauvé: {output_dir}")
        print(f"🎯 Si ça marche bien, on peut augmenter dataset et epochs!")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ OOM avec batch_size=8")
            print(f"💡 Relancez avec batch_size=4")
        else:
            print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()