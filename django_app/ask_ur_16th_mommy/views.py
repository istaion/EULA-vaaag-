import os
import time
from pathlib import Path
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import TemplateView, ListView, DeleteView
from ask_ur_16th_mommy.forms import TranslationForm, TranslationModelForm
from controller.controllers import call_groq_chat
from django.http import JsonResponse, HttpResponse
from django.urls import reverse, reverse_lazy
from django.contrib import messages
from django.conf import settings
from ask_ur_16th_mommy.models import Translation
import httpx
import sys
import base64
import subprocess
import tempfile

class HomeView(TemplateView):
    template_name = 'home.html'
    
    def get_available_voices(self): 
        voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../controller/voices'))
        if os.path.exists(voices_dir):
            return [f[:-4] for f in os.listdir(voices_dir) if f.lower().endswith(".wav")]
        else:
            return []
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = TranslationForm()
        context['available_voices'] = self.get_available_voices()
        context['translated_text'] = self.request.session.get('translated_text', '')
        context['selected_voice'] = self.request.session.get('selected_voice', '')
        audio_file = self.request.session.get('audio_file', '')

        # Vérifie l'existence réelle du fichier sur le disque
        if audio_file:
            abs_audio_path = os.path.join(settings.MEDIA_ROOT, audio_file.replace('/media/', ''))
            if not os.path.exists(abs_audio_path):
                self.request.session['audio_file'] = ''  # On nettoie la session
                audio_file = ''  # Plus d'audio file dans le contexte non plus

        context['audio_file'] = audio_file
        return context
    
    def post(self, request, *args, **kwargs):
        form = TranslationForm(request.POST)
        context = self.get_context_data(**kwargs)
        
        if form.is_valid():
            text_input = form.cleaned_data['text_input']
            translation_model = form.cleaned_data['translation_model']
            translated_text = self.translate_text(text_input, translation_model)
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
        from ask_ur_16th_mommy.models import Translation
        new_translation = Translation(text_to_translate=text)
        if model == "mBart":
            new_translation.model_type = "mBart"
            response = httpx.post(
                # "http://fastapi:8000/translate_mbart", 
                "http://localhost:8000/translate_mbart", 
                json={"text": text},
                timeout=30.0
            )
            if response.status_code != 200:
                raise RuntimeError(f"Erreur de traduction mBart: {response.text}")
            text = response.json()["translation"]
        elif model == "opus":
            new_translation.model_type = "opus"
            response = httpx.post(
                # "http://fastapi:8000/translate", 
                "http://localhost:8000/translate", 
                json={"text": text},
                timeout=30.0
            )
            if response.status_code != 200:
                raise RuntimeError(f"Erreur de traduction opus: {response.text}")
            text = response.json()["translation"]
        else:
            new_translation.model_type = "API"
            text = call_groq_chat(text)
        new_translation.translate_text = text
        new_translation.save()
        return text
    


class GenerateAudioView(TemplateView):
    template_name = 'home.html'
    
    def post(self, request, *args, **kwargs):
        translated_text = request.POST.get('translated_text', '')
        voice_selection = request.POST.get('voice_selection', '')
        
        if not translated_text or not voice_selection:
            messages.error(request, "Texte ou voix manquant.")
            return redirect('home')
        
        try:
            # audio_dir = os.path.join('media', 'audio')
            # audio_dir = os.path.join('media', 'audio')
            audio_dir = os.path.join(settings.MEDIA_ROOT, 'audio')
            # audio_dir = os.path.join(settings.MEDIA_ROOT, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            file_name = f"audio_{int(time.time())}"

            audio_path = self.call_worker_generate_audio(
                translated_text, audio_dir, file_name, voice_selection
            )
            
            if audio_path and os.path.exists(audio_path):
                request.session['translated_text'] = translated_text
                request.session['selected_voice'] = voice_selection
                request.session['audio_file'] = f"/media/audio/{file_name}.wav"
                messages.success(request, "Audio généré avec succès !")
            else:
                messages.error(request, "Erreur lors de la génération de l'audio.")
        except Exception as e:
            messages.error(request, f"Erreur lors de la génération : {str(e)}")
        return redirect('home')

    def call_worker_generate_audio(self, text, out_path, file_name, voice):
        """
        Appelle le worker TTS via subprocess pour garantir le vidage mémoire GPU.
        """
        import shlex

        # Chemin absolu du script worker
        worker_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../controller/worker_generate_audio.py')
        )
        args = [
            sys.executable, worker_script, text, out_path, file_name, voice
        ]
        # Pour éviter les problèmes d'espaces ou de caractères spéciaux, tu peux écrire "text" dans un fichier temporaire, si besoin.
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=180  # Peut augmenter si la génération prend du temps
            )
            if result.returncode != 0:
                print("stderr:", result.stderr)
                return None
            output_path = result.stdout.strip()
            return output_path
        except Exception as e:
            print(f"Erreur worker_generate_audio: {e}")
            return None


class VoiceRecordingView(TemplateView):
    template_name = 'voice_recording.html'
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context


class SaveRecordingView(TemplateView):
    def post(self, request, *args, **kwargs):
        audio_data = request.POST.get('audio_data', '')
        voice_name = request.POST.get('voice_name', '').strip()
        if not audio_data or not voice_name:
            messages.error(request, "Données audio ou nom de voix manquant.")
            return redirect('voice_recording')
        safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_voice_name = safe_voice_name.replace(' ', '_')
        audio_bytes = base64.b64decode(audio_data)
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../controller/voices'))
        os.makedirs(voices_dir, exist_ok=True)
        result_message = self.extract_sound_sample(temp_path, os.path.join(voices_dir, safe_voice_name))
        os.unlink(temp_path)
        if "succès" in result_message:
            messages.success(request, f"Voix '{voice_name}' enregistrée avec succès !")
        else:
            messages.error(request, f"Erreur : {result_message}")
        return redirect('voice_recording')

    def extract_sound_sample(self, sound_path, save_name):
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
    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get('audio_file')
        voice_name = request.POST.get('voice_name', '').strip()
        if not audio_file or not voice_name:
            messages.error(request, "Fichier audio ou nom de voix manquant.")
            return redirect('voice_recording')
        safe_voice_name = "".join(c for c in voice_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_voice_name = safe_voice_name.replace(' ', '_')
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as temp_file:
            for chunk in audio_file.chunks():
                temp_file.write(chunk)
            temp_path = temp_file.name
        voices_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../controller/voices'))
        os.makedirs(voices_dir, exist_ok=True)
        result_message = self.extract_sound_sample(temp_path, os.path.join(voices_dir, safe_voice_name))
        os.unlink(temp_path)
        if "succès" in result_message:
            messages.success(request, f"Voix '{voice_name}' traitée et sauvegardée avec succès !")
        else:
            messages.error(request, f"Erreur : {result_message}")
        return redirect('voice_recording')

    def extract_sound_sample(self, sound_path, save_name):
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
    model = Translation
    template_name = 'translation_list.html'
    context_object_name = 'translations'
    ordering = ['-id']
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_subtitle'] = "Historique de vos correspondances"
        return context

class TranslationDeleteView(DeleteView):
    model = Translation
    success_url = reverse_lazy('translation_list')
    def delete(self, request, *args, **kwargs):
        messages.success(request, "La traduction a été supprimée avec succès.")
        return super().delete(request, *args, **kwargs)
