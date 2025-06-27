from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import subprocess

router = APIRouter()

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    input_text: str
    translation: str

@router.post("/translate", response_model=TranslationResponse, summary="Traduire un texte via le modèle MarianMT", description="Reçoit un JSON `{ 'text': ... }` et renvoie la traduction.")
async def translate(request: TranslationRequest):
    try:
        result = subprocess.run(
            ["python", "controller/worker_translate_marian_mt.py", request.text],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        translation = result.stdout.strip().strip('"')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne de traduction: {e}")

    return TranslationResponse(
        input_text=request.text,
        translation=translation
    )

@router.post(
    "/translate_mbart",
    response_model=TranslationResponse,
    summary="Traduire un texte via le modèle mBART",
    description="Reçoit un JSON `{ 'text': ... }` et renvoie la traduction avec le modèle mBART fine-tuné."
)
async def translate_mbart(request: TranslationRequest):
    try:
        result = subprocess.run(
            ["python", "controller/worker_translate_mbart.py", request.text],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        translation = result.stdout.strip().strip('"')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne de traduction mBART: {e}")

    return TranslationResponse(
        input_text=request.text,
        translation=translation
    )
