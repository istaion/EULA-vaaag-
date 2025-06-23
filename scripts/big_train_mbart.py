import os
import torch
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import shutil

# ========== CONFIG ==========
MODEL_NAME = "facebook/mbart-large-50"
MAX_LEN = 128
BATCH_SIZE = 2
EPOCHS = 5
LR = 5e-5
OUTPUT_DIR = f"/kaggle/working/mbart_model_{datetime.now().strftime('%H%M')}"
DATASET_PATH = "notebooks/Vic/cleaned_data.csv"

# ========== GPU Check ==========
if not torch.cuda.is_available():
    raise EnvironmentError("🚨 GPU non disponible ! Activez le GPU T4 dans les paramètres Kaggle.")
print("✅ GPU disponible :", torch.cuda.get_device_name(0))

# ========== Dataset Load ==========
df = pd.read_csv(DATASET_PATH, sep="|")
df = df.iloc[:, :2]
df.columns = ['modern', 'old_french']
df = df.dropna()
df['modern'] = df['modern'].astype(str).str.strip()
df['old_french'] = df['old_french'].astype(str).str.strip()

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"📊 Dataset: train={len(train_df)}, val={len(val_df)}")

# ========== Dataset Class ==========
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, tokenizer, max_length):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        src = self.source_texts[idx]
        tgt = self.target_texts[idx]

        # Tokenisation séparée
        inputs = self.tokenizer(
            src,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()

        # Ignorer les tokens PAD dans les labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ========== Tokenizer / Model ==========
tokenizer = MBart50Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = "fr_XX"
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.config.forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"]
model.to("cuda")

# ========== DataSets ==========
train_dataset = TranslationDataset(train_df['modern'].tolist(), train_df['old_french'].tolist(), tokenizer, MAX_LEN)
val_dataset = TranslationDataset(val_df['modern'].tolist(), val_df['old_french'].tolist(), tokenizer, MAX_LEN)

# ========== Training Args ==========
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    save_total_limit=1,
    learning_rate=LR,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",
    fp16=True,
    gradient_checkpointing=True
)

# ========== Trainer ==========
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

print("\n🚀 Lancement de l'entraînement MBART...")
trainer.train()
print("\n✅ Entraînement terminé.")

# ========== Sauvegarde ==========
print(f"\n💾 Sauvegarde du modèle dans : {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ========== Génération Test ==========
print("\n🧪 Test de génération:")
test_input = "Bonjour, comment allez-vous ?"
inputs = tokenizer(test_input, return_tensors="pt", max_length=MAX_LEN, truncation=True).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=MAX_LEN,
        num_beams=4,
        forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
    )
    print("📝 Résultat:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# === COMPRESSION DU DOSSIER DU MODÈLE ===
zip_path = f"{OUTPUT_DIR}.zip"
print(f"📦 Compression du modèle en : {zip_path}")
shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)

# === AFFICHER LIEN DE TÉLÉCHARGEMENT ===
print(f"📥 Vous pouvez télécharger le modèle ici :")
print(f"/kaggle/working/{os.path.basename(zip_path)}")