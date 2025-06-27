import os
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView, ListView, DeleteView
from ask_ur_16th_mommy.models import Translation
from ask_ur_16th_mommy.forms import TranslationForm, TranslationModelForm
from ask_ur_16th_mommy.controller.controllers import call_groq_chat
from django.http import JsonResponse, HttpResponse
from django.urls import reverse, reverse_lazy
from django.contrib import messages
import httpx
import sys
import base64
import subprocess
import tempfile

class HomeView(TemplateView):
    template_name = 'home.html'
    
    def get_available_voices(self): 
        """Récupère la liste des voix disponibles"""
        voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ask_ur_16th_mommy', 'controller', 'voices')
        print(f"Recherche des voix dans : {voices_dir}")
        if os.path.exists(voices_dir):
            voices = [f[:-4] for f in os.listdir(voices_dir) if f.lower().endswith(".wav")]
            print(f"Voix trouvées : {voices}")
            return voices
        else:
            print("Dossier voices non trouvé")
            return []
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = TranslationForm()
        context['available_voices'] = self.get_available_voices()
        
        # Récupérer les données de la session si elles existent
        context['translated_text'] = self.request.session.get('translated_text', '')
        context['selected_voice'] = self.request.session.get('selected_voice', '')
        context['audio_file'] = self.request.session.get('audio_file', '')
        
        return context
    
    def post(self, request, *args, **kwargs):
        form = TranslationForm(request.POST)
        context = self.get_context_data(**kwargs)
        
        if form.is_valid():
            text_input = form.cleaned_data['text_input']
            translation_model = form.cleaned_data['translation_model']
            
            translated_text = self.translate_text(text_input, translation_model)
            
            # Stocker dans la session pour la génération audio
            request.session['translated_text'] = translated_text
            request.session['text_input'] = text_input
            request.session['translation_model'] = translation_model
            
            context.update({
                'form': form,
                'text_input': text_input,
                'translation_model': translation_model,
                'translated_text': translated_text,
                'available_voices': self.get_available_voices()
            })
        else:
            context['form'] = form
            
        return self.render_to_response(context)
    
    def translate_text(self, text, model):
        """
        Fonction de traduction utilisant votre logique existante
        """
        new_translation = Translation(text_to_translate=text)
        # clear_memory()
        if model == "mBart":
            new_translation.model_type = "mBart"
            response = httpx.post(
                # "http://localhost:8000/translate_mbart", 
                "http://fastapi:8000/translate_mbart", 
                json={"text": text},
                timeout=10.0
            )
            if response.status_code != 200:
                raise RuntimeError(f"Erreur de traduction OPUS: {response.text}")
            text = response.json()["translation"]
        elif model == "opus":
            new_translation.model_type = "opus"
            response = httpx.post(
                # "http://localhost:8000/translate", 
                "http://fastapi:8000/translate", 
                json={"text": text},
                timeout=10.0
            )
            if response.status_code != 200:
                raise RuntimeError(f"Erreur de traduction OPUS: {response.text}")
            text = response.json()["translation"]
        else :
            new_translation.model_type = "API"
            text = call_groq_chat(text)
        new_translation.translate_text = text
        new_translation.save()
        return text


class GenerateAudioView(TemplateView):
    template_name = 'home.html'

    def post(self, request, *args, **kwargs):
        print("=== POST GenerateAudioView ===")
        translated_text = request.POST.get('translated_text', '')
        voice_selection = request.POST.get('voice_selection', '')

        print(f"Translated text: {translated_text[:50]}...")
        print(f"Voice selection: {voice_selection}")

        if not translated_text or not voice_selection:
            messages.error(request, "Texte ou voix manquant.")
            return redirect('home')

        try:
            import time
            audio_dir = os.path.join('media', 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            file_name = f"audio_{int(time.time())}"
            output_path = os.path.join(audio_dir, f"{file_name}.wav")

            # Chemin absolu du script worker
            worker_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'controller', 'worker_audio_generation.py'
            )
            # Chemin absolu de la voix (adapte selon ton arbo)
            voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ask_ur_16th_mommy', 'controller', 'voices')
            voice_path = os.path.join(voices_dir, f"{voice_selection}.wav")

            # Appel du worker en sous-processus, SANS charger le modèle dans ce process
            result = subprocess.run(
                [
                    sys.executable,
                    worker_script,
                    translated_text,
                    output_path,
                    voice_path
                ],
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode != 0:
                print("stderr:", result.stderr)
                messages.error(request, f"Erreur génération audio : {result.stderr}")
                return redirect('home')

            output = result.stdout.strip()
            print("Sortie du worker :", output)
            if os.path.exists(output_path):
                request.session['translated_text'] = translated_text
                request.session['selected_voice'] = voice_selection
                request.session['audio_file'] = f"/media/audio/{file_name}.wav"
                messages.success(request, "Audio généré avec succès !")
            else:
                messages.error(request, "Erreur lors de la génération de l'audio.")
        except Exception as e:
            messages.error(request, f"Erreur lors de la génération : {str(e)}")
            print(f"Exception: {e}")

        return redirect('home')


class VoiceRecordingView(TemplateView):
    template_name = 'voice_recording.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class SaveRecordingView(TemplateView):
    """Vue pour sauvegarder un enregistrement direct depuis le navigateur"""
    
    def post(self, request, *args, **kwargs):
        print("=== SaveRecordingView POST ===")
        
        try:
            audio_data = request.POST.get('audio_data', '')
            voice_name = request.POST.get('voice_name', '').strip()
            
            if not audio_data or not voice_name:
                messages.error(request, "Données audio ou nom de voix manquant.")
                return redirect('voice_recording')
            
            # Nettoyer le nom de fichier
            safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_voice_name = safe_voice_name.replace(' ', '_')
            
            # Décoder les données base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            print(f"Fichier temporaire créé : {temp_path}")
            
            # Traiter l'audio avec ffmpeg
            voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ask_ur_16th_mommy', 'controller', 'voices')
            os.makedirs(voices_dir, exist_ok=True)
            
            result_message = self.extract_sound_sample(temp_path, os.path.join(voices_dir, safe_voice_name))
            
            # Supprimer le fichier temporaire
            os.unlink(temp_path)
            
            if "succès" in result_message:
                messages.success(request, f"Voix '{voice_name}' enregistrée avec succès !")
            else:
                messages.error(request, f"Erreur : {result_message}")
                
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")
            messages.error(request, f"Erreur lors de la sauvegarde : {str(e)}")
        
        return redirect('voice_recording')
    
    def extract_sound_sample(self, sound_path, save_name):
        """
        Votre fonction d'extraction adaptée
        """
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


class UploadVoiceView(TemplateView):
    """Vue pour traiter un fichier audio uploadé"""
    
    def post(self, request, *args, **kwargs):
        print("=== UploadVoiceView POST ===")
        
        try:
            audio_file = request.FILES.get('audio_file')
            voice_name = request.POST.get('voice_name', '').strip()
            
            if not audio_file or not voice_name:
                messages.error(request, "Fichier audio ou nom de voix manquant.")
                return redirect('voice_recording')
            
            # Nettoyer le nom de fichier
            safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_voice_name = safe_voice_name.replace(' ', '_')
            
            # Sauvegarder temporairement le fichier uploadé
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            print(f"Fichier uploadé sauvegardé temporairement : {temp_path}")
            
            # Traiter l'audio avec ffmpeg
            voices_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ask_ur_16th_mommy', 'controller', 'voices')
            os.makedirs(voices_dir, exist_ok=True)
            
            result_message = self.extract_sound_sample(temp_path, os.path.join(voices_dir, safe_voice_name))
            
            # Supprimer le fichier temporaire
            os.unlink(temp_path)
            
            if "succès" in result_message:
                messages.success(request, f"Voix '{voice_name}' traitée et sauvegardée avec succès !")
            else:
                messages.error(request, f"Erreur : {result_message}")
                
        except Exception as e:
            print(f"Erreur lors du traitement : {e}")
            messages.error(request, f"Erreur lors du traitement : {str(e)}")
        
        return redirect('voice_recording')
    
    def extract_sound_sample(self, sound_path, save_name):
        """
        Votre fonction d'extraction adaptée
        """
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


class TranslationListView(ListView):
    """Vue pour afficher la liste des traductions"""
    model = Translation
    template_name = 'translation_list.html'
    context_object_name = 'translations'
    ordering = ['-id']  # Ordre décroissant par ID (plus récent en premier)
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_subtitle'] = "Historique de vos correspondances"
        return context

class TranslationDeleteView(DeleteView):
    """Vue pour supprimer une traduction"""
    model = Translation
    success_url = reverse_lazy('translation_list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, "La traduction a été supprimée avec succès.")
        return super().delete(request, *args, **kwargs)