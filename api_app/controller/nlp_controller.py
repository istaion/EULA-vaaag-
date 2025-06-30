import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import os



def generate_audio(text, out_path, file_name, voice):
    # Autoriser la classe personnalisée pour le chargement sécurisé
    torch.serialization.add_safe_globals([XttsConfig])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)

    # Chemin vers le fichier de voix sélectionné
    voice_path = f"controller/voices/{voice}.wav"
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Fichier de voix non trouvé : {voice_path}")

    output_path = f"{out_path}/{file_name}.wav"
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=voice_path,
        language="fr"
    )
    return output_path

def clear_memory():
    """Nettoyage simple mémoire GPU."""
    import gc
    import torch
    torch.cuda.empty_cache()
    gc.collect()

