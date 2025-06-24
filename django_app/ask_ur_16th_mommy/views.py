from django.shortcuts import render
from django.views.generic import TemplateView
from ask_ur_16th_mommy.forms import TranslationForm
from ask_ur_16th_mommy.controllers import load_mbart_model, translate_with_mbart

class HomeView(TemplateView):
    template_name = 'home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = TranslationForm()
        return context
    
    def post(self, request, *args, **kwargs):
        form = TranslationForm(request.POST)
        context = self.get_context_data(**kwargs)
        
        if form.is_valid():
            text_input = form.cleaned_data['text_input']
            translation_model = form.cleaned_data['translation_model']
            
            # Ici vous pouvez intégrer votre logique de traduction
            # Par exemple avec une API ou un modèle de ML
            translated_text = self.translate_text(text_input, translation_model)
            
            context.update({
                'form': form,
                'text_input': text_input,
                'translation_model': translation_model,
                'translated_text': translated_text
            })
        else:
            context['form'] = form
            
        return self.render_to_response(context)
    
    def translate_text(self, text, model):
        """
        Fonction de traduction - à remplacer par votre logique métier
        """
        if model == "mBart":
            text = translate_with_mbart("../scripts/mbart_fast_1339", text)["text"]
        return text

