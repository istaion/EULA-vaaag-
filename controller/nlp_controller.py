import torch
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import os
import subprocess
from playsound import playsound




# Chargement du modèle et du tokenizer fine-tunés
def load_model(model_path = "model/marianmt-vieux-francais-model2") :

    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    return model, tokenizer


# fonction de traduction
def generate_translation_marian(text, model, tokenizer, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", max_length=128, truncation=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        gen = model.generate(**batch, max_length=400, num_beams=4)
    return tokenizer.decode(gen[0], skip_special_tokens=True)


# générer l'audio
def generate_audio(text, out_path, file_name) :

    voices_list = [f[:-4] for f in os.listdir("controller/voices") if f.lower().endswith(".wav")]
    for i, v in enumerate(voices_list):
        print(f"{i+1} : {v}")
    try:
        choice = int(input("Choisissez la voix (numéro) : "))
        voice = voices_list[choice - 1]
    except (ValueError, IndexError):
        print("Entrée invalide, veuillez saisir un numéro parmi la liste.")
        return None

    # Autoriser la classe personnalisée pour le chargement sécurisé
    torch.serialization.add_safe_globals([XttsConfig])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)

    tts.tts_to_file(
        text=text,
        file_path=f"{out_path}/{file_name}.wav",
        speaker_wav=f"controller/voices/{voice}.wav",
        language="fr"
    )
    return f"{out_path}/{file_name}.wav"


# lire le fichier audio généré
def read_audio(audio_path) :
    playsound(audio_path)




# read_audio("cartman.wav")


    # modern_dir = "modern_fr_text_sound"
    # old_dir = "old_fr_text_sound"

    # subprocess.run(f"rm -rf {modern_dir}/*", shell=True)
    # subprocess.run(f"rm -rf {old_dir}/*", shell=True)

    # modern_audio = generate_audio(modern_text, modern_dir, "modern_sound")
    # old_audio = generate_audio(old_text, old_dir, "old_sound")



