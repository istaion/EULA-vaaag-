#!/usr/bin/env python3
import sys
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python worker_audio_generation.py <text> <output_path> <voice_path>", file=sys.stderr)
        sys.exit(1)
    
    text = sys.argv[1]
    output_path = sys.argv[2]
    voice_path = sys.argv[3]
    
    try:
        import torch
        from TTS.api import TTS
        from TTS.tts.configs.xtts_config import XttsConfig
        
        # Autoriser la classe personnalisée pour le chargement sécurisé
        torch.serialization.add_safe_globals([XttsConfig])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialiser le modèle TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        
        # Vérifier que le fichier de voix existe
        if not os.path.exists(voice_path):
            print(f"Fichier de voix non trouvé : {voice_path}", file=sys.stderr)
            sys.exit(1)
        
        # Générer l'audio
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=voice_path,
            language="fr"
        )
        
        # Vérifier que le fichier a été créé
        if os.path.exists(output_path):
            print(output_path)  # Retourner le chemin du fichier généré
            sys.exit(0)
        else:
            print("Échec de la génération du fichier audio", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Erreur lors de la génération audio : {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
