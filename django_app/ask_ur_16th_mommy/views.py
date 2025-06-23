from django.shortcuts import render
from django.views.generic import TemplateView
from ask_ur_16th_mommy.forms import TranslationForm

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
        # Exemples de transformations basiques selon le modèle choisi
        transformations = {
            'medieval': {
                'vous': 'vos',
                'je': 'jeo',
                'mon': 'mien',
                'ma': 'meie',
                'avec': 'ovec',
                'très': 'molt',
                'grand': 'grant',
                'grande': 'grante',
                'maintenant': 'ore',
                'aujourd\'hui': 'hui',
                'maison': 'meson',
            },
            'renaissance': {
                'vous': 'vous',
                'je': 'ie',
                'mon': 'mon',
                'très': 'fort',
                'maintenant': 'à présent',
                'aujourd\'hui': 'auiourd\'huy',
                'maison': 'logis',
            },
            'classical': {
                'très': 'fort',
                'maintenant': 'présentement',
                'aujourd\'hui': 'auiourd\'huy',
                'avec': 'avecques',
            },
            'old_french': {
                'vous': 'vus',
                'je': 'eo',
                'mon': 'meon',
                'ma': 'meie',
                'avec': 'od',
                'très': 'mult',
                'maintenant': 'or',
            }
        }
        
        if model in transformations:
            translated = text.lower()
            for modern, old in transformations[model].items():
                translated = translated.replace(modern, old)
            
            # Ajouter quelques formules de politesse d'époque
            if model == 'medieval':
                prefix = "Tres chiere et honouree dame, "
                suffix = "\n\nVostre humble serviteur."
            elif model == 'renaissance':
                prefix = "Madame ma tres honoree, "
                suffix = "\n\nVostre tres humble et obeissant serviteur."
            elif model == 'classical':
                prefix = "Madame, "
                suffix = "\n\nJ'ay l'honneur d'estre, Madame, vostre tres humble serviteur."
            else:  # old_french
                prefix = "Mult chiere dame, "
                suffix = "\n\nLi vostres hom."
                
            return f"{prefix}{translated.capitalize()}{suffix}"
        
        return text

