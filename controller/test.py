import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# Autoriser la classe personnalisée pour le chargement sécurisé
torch.serialization.add_safe_globals([XttsConfig])

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(device)


# Synthèse vocale avec clonage
text = "Par Diex ! Uncore une feiz, j'ai faillie moi cheoir en me hastant por trover les latrines sur le char.  J'avoie trop dolour al ventre, j'estoie just que j'allerai getter mon petit disner, j'estoie tant angoissée sur la voie. JEANNE, AU SECOURS ! Miam"

tts.tts_to_file(
    text=text,
    file_path="cartman.wav",
    speaker_wav="controller/voices/cartman.wav",
    language="fr"
)


