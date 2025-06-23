from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from controller.nlp_controller import load_model, generate_translation_marian


router = APIRouter()

class TranslationRequest(BaseModel):
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


