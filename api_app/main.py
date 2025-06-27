from fastapi import FastAPI
from endpoints.translate_endpoint import router as translate_router

# Créer l'application FastAPI
app = FastAPI(
    title="API de traduction historique",
    description=(
        "API utilisant différents modèles pour traduire "
        "des textes du français moderne vers le français "
        "avant le XVIᵉ siècle"
    ),
    version="1.0.0"
)

# Monter le routeur de traduction sous /translate
app.include_router(translate_router, tags=["traduction"])

@app.get("/", summary="Point d'entrée", tags=["root"])
async def root():
    """
    Endpoint racine pour vérifier que l'API fonctionne.
    """
    return {
        "message": "API de traduction historique active",
        "status": "ok",
        "endpoints": {
            "Traduction": "/translate",
            "Swagger UI": "/docs",
            "ReDoc": "/redoc"
        }
    }
