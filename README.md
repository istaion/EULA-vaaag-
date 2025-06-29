
# 🏰 EULA-VAAAG — Traduction & Synthèse Vocale de Vieux Français

## Dépendances principales

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2.3-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.13-lightgrey.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0+-f7941d.svg)
![torch](https://img.shields.io/badge/Torch-2.0.0+-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0+-ff6f00.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.37.2-9c27b0.svg)
![TTS](https://img.shields.io/badge/TTS-0.22.0+-blueviolet.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-1.88.0+-black.svg)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.34.3+-informational.svg)
![Gunicorn](https://img.shields.io/badge/Gunicorn-23.0.0+-brightgreen.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.34.1+-blue.svg)
![HuggingFace Hub](https://img.shields.io/badge/HuggingFace--Hub-0.33.1+-yellow.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.9.1+-yellowgreen.svg)
![matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3+-red.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2+-blue.svg)
![Datasets](https://img.shields.io/badge/Datasets-3.6.0+-teal.svg)
![SentencePiece](https://img.shields.io/badge/SentencePiece-0.2.0+-purple.svg)
![Pydantic](https://img.shields.io/badge/Pydantic-2.11.7+-lightblue.svg)
![dotenv](https://img.shields.io/badge/dotenv-0.9.9+-black.svg)
![sacrebleu](https://img.shields.io/badge/Sacrebleu-2.5.1+-purple.svg)
![rouge-score](https://img.shields.io/badge/Rouge--Score-0.1.2+-pink.svg)
![pyarrow](https://img.shields.io/badge/PyArrow-20.0.0+-lightgrey.svg)
![ipykernel](https://img.shields.io/badge/IPyKernel-6.29.5+-orange.svg)
![sacremoses](https://img.shields.io/badge/Sacremoses-0.1.1+-brown.svg)
![tf-keras](https://img.shields.io/badge/tf--keras-2.19.0+-magenta.svg)

## Description

Projet complet pour la **traduction automatique** du français moderne vers le vieux français, avec **synthèse vocale personnalisable**.

Comprend : API REST FastAPI, interface web Django, gestion et génération de voix (XTTS-v2), entraînement custom des modèles, déploiement Dockerisé avec Nginx/HTTPS.

---

## Sommaire

- [🏰 EULA-VAAAG — Traduction \& Synthèse Vocale de Vieux Français](#-eula-vaaag--traduction--synthèse-vocale-de-vieux-français)
  - [Dépendances principales](#dépendances-principales)
  - [Description](#description)
  - [Sommaire](#sommaire)
  - [Fonctionnalités](#fonctionnalités)
  - [Architecture du projet](#architecture-du-projet)
  - [Installation \& Déploiement](#installation--déploiement)
    - [Prérequis](#prérequis)
    - [1. Installation initiale (Clone \& build)](#1-installation-initiale-clone--build)
    - [1.1. Lancement courant (après installation)](#11-lancement-courant-après-installation)
    - [2. Accès](#2-accès)
    - [3. Variables / Configuration](#3-variables--configuration)
  - [Utilisation](#utilisation)
    - [Application Web](#application-web)
    - [API REST (FastAPI)](#api-rest-fastapi)
      - [Traduction (MarianMT ou mBART) :](#traduction-marianmt-ou-mbart)
      - [Gestion des voix audio](#gestion-des-voix-audio)
      - [Extraction d’un échantillon audio](#extraction-dun-échantillon-audio)
      - [Entraînement \& Données](#entraînement--données)
      - [Configuration Docker \& Nginx (HTTPS)](#configuration-docker--nginx-https)
  - [Développement local](#développement-local)
    - [1. Lancer l’API FastAPI](#1-lancer-lapi-fastapi)
    - [2. Migration / static (pour dev)](#2-migration--static-pour-dev)
    - [3. Lancer Django (doit attendre que l’API soit up !)](#3-lancer-django-doit-attendre-que-lapi-soit-up-)
  - [Contribuer](#contribuer)
  - [Licence](#licence)
  - [Remarques](#remarques)
  - [Crédits](#crédits)

---

## Fonctionnalités

- **Traduction automatique** (moderne → vieux français) via MarianMT et mBART fine-tunés
- **Synthèse vocale** : audio `.wav` avec choix de voix custom (XTTS-v2)
- **Upload / enregistrement de voix** utilisateur
- **Historique des traductions et audios**
- **API REST FastAPI** pour intégration externe ou batch
- **Interface web Django** responsive et complète
- **Déploiement sécurisé** (Docker Compose, Nginx, HTTPS)
- **Notebooks d’entraînement** et scripts pour data science

---

## Architecture du projet

```
├── api_app/ # API FastAPI pour la traduction (REST)
│ ├── controller/ # NLP/ML : workers, models, voice management
│ ├── endpoints/ # Routes FastAPI (translate)
│ └── model/ # Modèles MarianMT, mBART fine-tunés
├── create_data/ # Génération et préparation de jeux de données
├── data/ # Corpus, datasets, xml, csv nettoyés
├── django_app/ # Application Django (frontend + TTS)
│ ├── ask_ur_16th_mommy/ # App principale Django (UI, views, models)
│ │ ├── controller/voices/ # Voix custom .wav
│ │ ├── templates/ # Templates HTML Django
│ │ └── views.py # Logique UI
│ ├── media/audio/ # Audios générés (.wav)
│ └── model/xtts_v2/ # Modèle XTTS-v2 pour TTS
├── notebooks/ # Jupyter notebooks (prétraitement, entraînement)
├── scripts/ # Training/test sur modèles et data
├── nginx/ # Nginx config + certificats (reverse-proxy HTTPS)
├── docker-compose.yaml # Orchestration multi-services
├── Dockerfile.django # Build Django (TTS, UI)
├── Dockerfile.fastapi # Build FastAPI (NLP)
├── django_entrypoint.sh # Entrée Django (migrations, static, gunicorn)
├── start_project.sh # Script de lancement (préparation + docker compose)
└── README.md # Ce fichier
```

---

## Installation & Déploiement

### Prérequis

- **Docker** & **Docker Compose** installés
- **huggingface_hub** installé
- (Recommandé) GPU NVIDIA pour accélération CUDA/TTS
- (Recommandé) Compte [Hugging Face](https://huggingface.co/) pour télécharger les modèles personnalisés

### 1. Installation initiale (Clone & build)

```bash
git clone https://github.com/istaion/EULA-vaaag-.git
cd EULA-vaaag-
chmod +x start_project.sh
./start_project.sh
```

### 1.1. Lancement courant (après installation)

```bash
docker compose up
```

Ce script :

    Télécharge automatiquement tous les modèles nécessaires depuis HuggingFace dans les bons dossiers

    Redémarre le service NVIDIA (si dispo)

    Lance le déploiement docker (API, Django, Nginx/HTTPS)

### 2. Accès

- **Django (web UI)** :  
  http://localhost:8001  
  ou http://<IP_DE_LA_MACHINE>:8001 pour accès réseau local (autres PC)

- **FastAPI (API REST)** :  
  http://localhost:8000/docs  
  ou http://<IP_DE_LA_MACHINE>:8000/docs

### 3. Variables / Configuration

- Les accès GPU sont automatiquement pris en compte (voir docker-compose.yaml)

- Les modèles sont automatiquement téléchargés et placés dans :

    - /api_app/model/marianmt-vieux-francais-model2 pour le modèle MarianMT (voir sur [Hugging Face](https://huggingface.co/malek-b/mon-marianmt-vieux-francais-model2/))

    - /api_app/model/mbart/mbart_model_0928 pour le modèle mBART (voir sur [Hugging Face](https://huggingface.co/malek-b/mbart_model_0928/))

    - /django_app/model/xtts_v2/ pour le modèle XTTS-v2 (XTTS-v2 sur [Hugging Face](https://huggingface.co/coqui/XTTS-v2))

- Les chemins sont prévus pour l'intégration directe dans les scripts/applications.
  
- **Clé API GROQ (pour la traduction via LLM)**

- Crée un fichier `.env` à la racine du projet
- Ajoute ta clé comme ceci :
    ```env
    GROQ_API_KEY=ton_token_groq_ici
    ```
---

## Utilisation

### Application Web

1. Accéder à l’URL (voir ci-dessus)
2. Entrer un texte moderne, choisir le modèle (MarianMT/mBART/OpenAI), cliquer "Traduire"
3. Générer l’audio avec la voix souhaitée (upload ou enregistrement possible)
4. Télécharger ou écouter le `.wav` dans l’UI
5. Historique disponible (suppression possible)

### API REST (FastAPI)

#### Traduction (MarianMT ou mBART) :

- `POST /translate`
- `POST /translate_mbart`
- Payload JSON : `{ "text": "Votre texte moderne" }`
- Réponse JSON : `{ "input_text": "...", "translation": "..." }`

Exemple avec cURL :
```bash
curl -X POST http://localhost:8000/translate   -H "Content-Type: application/json"   -d '{"text":"Bonjour, comment allez-vous ?"}'
```

#### Gestion des voix audio

- Dossier des voix : `django_app/ask_ur_16th_mommy/controller/voices/`
- Fichiers `.wav` mono, 24kHz, ~8s recommandés
- Ajout via interface (upload/enregistrement)
- Les voix sont utilisées par XTTS-v2 pour personnaliser la synthèse audio

#### Extraction d’un échantillon audio

Pour extraire un échantillon compatible :

```bash
ffmpeg -i chemin/vers/ton_audio.mp3(.wav) -ss 00:00:00 -t 8 -ac 1 -ar 24000 django_app/ask_ur_16th_mommy/controller/voices/nom_de_la_voix.wav
```

#### Entraînement & Données

- Notebooks : `notebooks/` (prétraitement, tokenisation, entraînement MarianMT, mBART, T5…)
- Données sources : `data/` (csv alignés, corpus XML, lemmatisation)
- Scripts : `create_data/`, `scripts/` (préparation data, entraînement batch)
- Nouveaux modèles : placer le dossier du modèle entraîné dans `api_app/model/` (pour la traduction), `django_app/model/xtts_v2/` pour TTS.


#### Configuration Docker & Nginx (HTTPS)

- Nginx reverse-proxy HTTPS/SSL (nginx/certs/, nginx.conf)

- Certificats SSL : auto-signés ou réels, à placer dans nginx/certs/

  - Génération auto-signée :

  ```bash
  mkdir -p nginx/certs
  openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout nginx/certs/selfsigned.key -out nginx/certs/selfsigned.crt -subj "/CN=localhost"
  ```

- Configuration Nginx personnalisée :

    - Adapter nginx/nginx.conf si besoin (exemple inclus)
    - Proxy HTTPS → Django/FastAPI (ports internes)

- Variables importantes :

    - STATIC_ROOT et MEDIA_ROOT bien configurés pour Django
    - ALLOWED_HOSTS et CSRF_TRUSTED_ORIGINS ouverts pour le LAN (en prod : restreindre !)


## Développement local

### 1. Lancer l’API FastAPI

```bash
cd api_app
uvicorn main:app --reload
```

### 2. Migration / static (pour dev)

```bash
cd django_app
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic
```

### 3. Lancer Django (doit attendre que l’API soit up !)

```bash
python manage.py runserver
```

---

## Contribuer

- Fork, branche, PRs bienvenues !
- Notebooks et scripts à commenter/documenter pour la communauté
- Pensez à fournir des exemples de nouvelles voix / modèles !

---

## Licence

MIT — Open Source, attribution requise.

---

## Remarques

- Les notebooks notebooks/Malek/ et notebooks/Vic/ contiennent toutes les étapes de preprocessing, d’expérimentation et d’entraînement des modèles.
- Le dossier media/audio/ reçoit les fichiers audios générés par l’UI Django.
- Les voix custom doivent être au format .wav, mono, 24kHz, max 8s.

---

## Crédits

Projet réalisé par :

<div>
<h4>Malek B. </h4>
  <a href="https://github.com/Malek-Boumedine" target="_blank">
  <img loading="lazy" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href = "mailto: malek.boumedine@gmail.com"><img loading="lazy" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" target="_blank">
  </a>
</div>

<div>
<h4>Victor P.</h4>
    <a href="https://www.github.com/istaion" target="_blank">
    <img loading="lazy" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white">
    <a href = "mailto: v.poutot@gmail.com"><img loading="lazy" src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" target="_blank"></a>
</div>

Pour toute question, contactez-nous ou ouvrez une issue GitHub !  
_Bon vieux français à tous_ 


[def]: #-eula-vaaag--traduction--synthèse-vocale-de-vieux-français