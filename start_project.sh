#!/bin/bash

set -e

# ---- CONFIGURATION ----
MARIAN_DIR="api_app/model/marianmt-vieux-francais-model2"
MBART_DIR="api_app/model/mbart/mbart_model_0928"

# ---- GÉNÉRATION DES CERTIFICATS SSL (si absents) ----
if [ ! -f nginx/certs/selfsigned.key ] || [ ! -f nginx/certs/selfsigned.crt ]; then
  echo "Génération d'un certificat SSL auto-signé pour Nginx..."
  mkdir -p nginx/certs
  openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
    -keyout nginx/certs/selfsigned.key \
    -out nginx/certs/selfsigned.crt \
    -subj "/CN=localhost"
else
  echo "Certificat SSL déjà présent, skip génération."
fi

# ---- Téléchargement modèle MarianMT ----
echo "Téléchargement du modèle MarianMT depuis Hugging Face..."
mkdir -p "$MARIAN_DIR"
# python3 -m huggingface_hub download malek-b/mon-marianmt-vieux-francais-model2 --local-dir "$MARIAN_DIR" --local-dir-use-symlinks False
huggingface-cli download malek-b/mon-marianmt-vieux-francais-model2 --local-dir "$MARIAN_DIR" --local-dir-use-symlinks False

# ---- Téléchargement modèle mBART ----
echo "Téléchargement du modèle mBART depuis Hugging Face..."
mkdir -p "$MBART_DIR"
# python3 -m huggingface_hub download malek-b/mbart_model_0928 --local-dir "$MBART_DIR" --local-dir-use-symlinks False
huggingface-cli download malek-b/mbart_model_0928 --local-dir "$MBART_DIR" --local-dir-use-symlinks False

# ---- Service NVIDIA (optionnel) ----
if systemctl list-units --full -all | grep -Fq 'nvidia-persistenced.service'; then
    echo "Redémarrage du service NVIDIA persistenced..."
    sudo systemctl restart nvidia-persistenced.service
else
    echo "Pas de service NVIDIA (aucun GPU NVIDIA détecté, skip)"
fi

# ---- Démarrage Docker ----
echo "Démarrage des containers Docker..."
docker compose up --build
