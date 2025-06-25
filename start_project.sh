#!/bin/bash

echo "Redémarrage du service NVIDIA persistenced..."
sudo systemctl restart nvidia-persistenced.service

echo "Démarrage des containers Docker..."
docker compose up --build
