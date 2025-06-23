from django import forms

class TranslationForm(forms.Form):
    TRANSLATION_CHOICES = [
        ('', 'Choisissez un modèle...'),
        ('mBart', 'mBart'),
        ('API', 'openAI API'),
        ('opus', 'Helsinki-NLP/opus-mt-fr-en')
    ]
    
    text_input = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-textarea',
            'placeholder': 'Écrivez ici le texte que vous souhaitez transformer en vieux français...',
            'rows': 10
        }),
        label='Votre texte à traduire :',
        required=True
    )
    
    translation_model = forms.ChoiceField(
        choices=TRANSLATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Modèle de traduction :',
        required=True
    )