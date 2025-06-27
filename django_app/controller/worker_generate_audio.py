import sys
import os
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from django.conf import settings

def main():
    if len(sys.argv) != 5:
        print("Usage: python worker_generate_audio.py <text> <out_path> <file_name> <voice>", file=sys.stderr)
        sys.exit(1)

    text, out_path, file_name, voice = sys.argv[1:5]

    # Chemin absolu pour la voix et le fichier de sortie
    voice_path = os.path.abspath(f"controller/voices/{voice}.wav")
    output_path = os.path.abspath(os.path.join(out_path, f"{file_name}.wav"))

    # Création du dossier cible si besoin
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Vérif voix
    if not os.path.exists(voice_path):
        print(f"Fichier de voix non trouvé : {voice_path}", file=sys.stderr)
        sys.exit(2)

    # Génération audio
    torch.serialization.add_safe_globals([XttsConfig])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)

    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=voice_path,
        language="fr"
    )

    # Sortie brute du chemin absolu pour Django
    print(output_path)
    # Après appel du worker_generate_audio
    print("Audio path (from worker):", output_path)
    print("os.path.exists(output_path):", os.path.exists(output_path))
    print("Contenu du dossier audio:", os.listdir(os.path.dirname(output_path)))

if __name__ == "__main__":
    main()
