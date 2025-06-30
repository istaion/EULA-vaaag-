from django.db import models

MODEL_CHOICES = {   
        'mBart': 'mBart',
        'API': 'API',
        'opus': 'opus'}

class Translation(models.Model):

    text_to_translate = models.TextField()
    translate_text = models.TextField()
    model_type = models.CharField(max_length=124, choices=MODEL_CHOICES)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
