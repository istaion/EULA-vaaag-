import torch
from transformers import MarianMTModel, MarianTokenizer, MBartForConditionalGeneration, MBart50Tokenizer
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import os
import subprocess
import gc
import sys
from playsound import playsound
import signal
import time
import psutil
from pathlib import Path




# Chargement du modèle et du tokenizer fine-tunés
def load_model(model_path = "model/marianmt-vieux-francais-model2") :

    # clear_cuda_memory()

    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    return model, tokenizer


# Chargement du modèle et du tokenizer fine-tunés mBART
def load_mbart_model(model_path="model/mbart/mbart_model_0928"):

    # clear_cuda_memory()
    
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50Tokenizer.from_pretrained(model_path)
    tokenizer.src_lang = "fr_XX"
    model.config.forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"]
    return model, tokenizer


# fonction de traduction mBART
# def generate_translation_mbart(text, model, tokenizer, device=None):
    
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
    
#     # Tokenisation avec mBART
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         max_length=1024,
#         truncation=True,
#         padding=True
#     )
#     inputs = {k: v.to(device) for k, v in inputs.items()}
    
#     with torch.no_grad():
#         generated_tokens = model.generate(
#             **inputs,
#             max_length=400,
#             num_beams=4,
#             forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
#         )
    
#     return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

def generate_translation_mbart(text, model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    try:
        # Tokenisation avec mBART
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=400,
                num_beams=4,
                forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
            )

        output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    finally:
        # Libération explicite des ressources GPU
        del model
        del tokenizer
        del inputs
        torch.cuda.empty_cache()
    
    return output


# fonction de traduction
# def generate_translation_marian(text, model, tokenizer, device=None):

#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()
#     batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", max_length=128, truncation=True)
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         gen = model.generate(**batch, max_length=400, num_beams=4)
#     return tokenizer.decode(gen[0], skip_special_tokens=True)

def generate_translation_marian(text, model, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch = tokenizer.prepare_seq2seq_batch(
        [text], return_tensors="pt", max_length=128, truncation=True
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        gen = model.generate(**batch, max_length=400, num_beams=4)

    output = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Nettoyage GPU
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return output


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


def extract_sound_sample(sound_path, save_name):
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", sound_path, "-ss", "00:00:01", "-t", "8", "-ac", "1", "-ar", "24000", f"{save_name}.wav"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            return "Échantillon enregistré avec succès !"
        else:
            return "Échec lors de l'enregistrement de l'échantillon"
    except FileNotFoundError:
        return "Erreur : ffmpeg n'est pas installé"
    



    # modern_dir = "modern_fr_text_sound"
    # old_dir = "old_fr_text_sound"

    # subprocess.run(f"rm -rf {modern_dir}/*", shell=True)
    # subprocess.run(f"rm -rf {old_dir}/*", shell=True)

    # modern_audio = generate_audio(modern_text, modern_dir, "modern_sound")
    # old_audio = generate_audio(old_text, old_dir, "old_sound")

def get_gpu_processes():
    """Récupère la liste des processus utilisant le GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        processes = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        processes.append({
                            'pid': int(parts[0]),
                            'name': parts[1],
                            'memory': int(parts[3])
                        })
        return processes
    except Exception as e:
        print(f"❌ Erreur récupération processus GPU: {e}")
        return []

def kill_python_gpu_processes():
    """Tue les processus Python utilisant le GPU"""
    print("🔍 Recherche des processus Python sur GPU...")
    gpu_processes = get_gpu_processes()
    
    killed = 0
    for proc in gpu_processes:
        if 'python' in proc['name'].lower():
            try:
                pid = proc['pid']
                memory = proc['memory']
                
                # Vérifier si le processus existe encore
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    print(f"🎯 Processus trouvé: PID {pid} - {proc['name']} ({memory}MB)")
                    print(f"   Commande: {' '.join(process.cmdline()[:3])}...")
                    
                    # Demander confirmation pour les gros processus
                    if memory > 1000:
                        response = "y"
                        if response.lower() != 'y':
                            continue
                    
                    # Tentative d'arrêt propre d'abord
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                        print(f"✅ Processus {pid} terminé proprement")
                    except psutil.TimeoutExpired:
                        # Forcer l'arrêt si nécessaire
                        process.kill()
                        print(f"⚡ Processus {pid} forcé à s'arrêter")
                    
                    killed += 1
                    time.sleep(1)  # Pause entre les kills
                    
            except Exception as e:
                print(f"❌ Erreur avec processus {proc['pid']}: {e}")
    
    print(f"📊 {killed} processus Python tués")
    return killed

def restart_nvidia_services():
    """Redémarre les services NVIDIA"""
    services = [
        'nvidia-persistenced.service',
        'nvidia-fabricmanager.service'  # Si disponible
    ]
    
    for service in services:
        try:
            print(f"🔄 Redémarrage {service}...")
            
            # Vérifier si le service existe
            check_cmd = ['systemctl', 'status', service]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if 'not found' not in check_result.stderr:
                # Arrêter le service
                subprocess.run(['sudo', 'systemctl', 'stop', service], check=True)
                time.sleep(2)
                
                # Redémarrer le service
                subprocess.run(['sudo', 'systemctl', 'start', service], check=True)
                print(f"✅ {service} redémarré")
            else:
                print(f"⚠️  {service} non trouvé")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur avec {service}: {e}")

def force_cuda_cleanup():
    """Nettoyage CUDA agressif"""
    try:
        import torch
        if torch.cuda.is_available():
            print("🧹 Nettoyage CUDA agressif...")
            
            # Vider tous les caches
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            
            # Forcer la libération mémoire sur tous les devices
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            
            print("✅ Nettoyage CUDA terminé")
    except ImportError:
        print("⚠️  PyTorch non disponible")

def reset_gpu_compute_mode():
    """Essaie de réinitialiser le mode compute du GPU"""
    try:
        print("🔧 Tentative reset mode compute...")
        # Essayer de réinitialiser sans affecter l'affichage
        result = subprocess.run(['sudo', 'nvidia-smi', '-c', '0'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Mode compute réinitialisé")
        else:
            print("⚠️  Reset mode compute échoué (normal pour GPU primaire)")
    except Exception as e:
        print(f"❌ Erreur reset compute: {e}")

def show_memory_status():
    """Affiche l'état de la mémoire GPU"""
    try:
        print("📊 État mémoire GPU:")
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            used, free, total = map(int, result.stdout.strip().split(', '))
            print(f"   Utilisée: {used}MB / {total}MB ({used/total*100:.1f}%)")
            print(f"   Libre: {free}MB")
    except Exception as e:
        print(f"❌ Erreur lecture mémoire: {e}")

def clear_memory():
    """Fonction principale"""
    print("🚀 Nettoyage GPU avancé - Version Python killer\n")
    
    # État initial
    show_memory_status()
    print()
    
    # # Nettoyage des processus Python
    # killed = kill_python_gpu_processes()
    
    # if killed > 0:
    #     print("\n⏳ Attente stabilisation...")
    #     time.sleep(3)
    
    # Nettoyage CUDA
    force_cuda_cleanup()
    
    # Garbage collection
    print("🗑️  Garbage collection...")
    collected = gc.collect()
    print(f"   {collected} objets Python supprimés")
    
    # Redémarrage services NVIDIA si demandé
    if '--restart-services' in sys.argv:
        print("\n🔄 Redémarrage services NVIDIA...")
        restart_nvidia_services()
        time.sleep(5)
    
    # Reset compute mode si demandé
    if '--reset-compute' in sys.argv:
        reset_gpu_compute_mode()
    
    print("\n" + "="*60)
    print("✨ Nettoyage terminé!")
    show_memory_status()
    
    if '--restart-services' not in sys.argv:
        print("\n💡 Ajoutez --restart-services pour redémarrer nvidia-persistenced")
    if '--reset-compute' not in sys.argv:
        print("💡 Ajoutez --reset-compute pour tenter un reset du mode compute")

