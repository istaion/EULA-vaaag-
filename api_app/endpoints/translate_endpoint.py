from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from controller.nlp_controller import load_model, generate_translation_marian, load_mbart_model, generate_translation_mbart


router = APIRouter()

class TranslationRequest(BaseModel):
    # text: f"""{str}"""
    text: str

class TranslationResponse(BaseModel):
    input_text: str
    translation: str

@router.post("/translate", response_model=TranslationResponse, summary="Traduire un texte via un model choisi", description="Reçoit un JSON `{ 'text': ... }` et renvoie la traduction.")
async def translate(request: TranslationRequest):    
    try:
        model, tokenizer = load_model()
        translation = generate_translation_marian(request.text, model, tokenizer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="erreur interne de traduction")

    return TranslationResponse(
        input_text=request.text,
        translation=translation
    )

@router.post("/translate_mbart", response_model=TranslationResponse, summary="Traduire un texte via le modèle mBART", description="Reçoit un JSON `{ 'text': ... }` et renvoie la traduction avec le modèle mBART fine-tuné.")
async def translate_mbart(request: TranslationRequest):    
    try:
        model, tokenizer = load_mbart_model()
        translation = generate_translation_mbart(request.text, model, tokenizer)
    except Exception as e:
        raise HTTPException(status_code=500, detail="erreur interne de traduction mBART")

    return TranslationResponse(
        input_text=request.text,
        translation=translation
    )
