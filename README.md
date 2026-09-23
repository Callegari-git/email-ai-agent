# Email Triage Agent

[![CI](https://github.com/Callegari-git/email-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Callegari-git/email-ai-agent/actions/workflows/ci.yml)

API FastAPI qui récupère les emails non lus (Gmail), les fait évaluer par Gemini
(importance 1 à 5 + résumé de 3 lignes max) et renvoie un JSON structuré, validé par Pydantic.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # puis renseigner GEMINI_API_KEY
```

Pour Gmail (`EMAIL_PROVIDER=gmail`) : placer le client OAuth « Application de bureau »
dans `credentials.json`, puis autoriser l'accès une fois :

```bash
python -m scripts.authorize_gmail
```

## Lancer l'API

```bash
uvicorn app.main:app --reload
```

Swagger : http://127.0.0.1:8000/docs

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | État du service (sans appel externe) |
| GET | `/emails/unread` | Emails non lus extraits, sans analyse (gratuit) |
| POST | `/emails/unread/analyze` | Récupère et analyse les non lus, triés par priorité |
| POST | `/emails/analyze` | Analyse un email fourni dans la requête (exemples dans Swagger) |

Erreurs : format commun `{"error": "<code>", "detail": "<message>"}`.

| Code HTTP | `error` | Cause |
|---|---|---|
| 503 | `email_auth_required` | Token Gmail absent/expiré : relancer `scripts.authorize_gmail` |
| 503 | `email_provider_unavailable`, `llm_unavailable` | Erreur transitoire, réessayer après `Retry-After` |
| 502 | `email_provider_error`, `llm_invalid_response`, `llm_error` | Réponse inattendue de Gmail ou Gemini |
| 500 | `llm_configuration_error` | Clé API, modèle ou paramètre Gemini invalide |

## Structure

```
app/
├── main.py                  # Création de l'app, lifespan (services partagés)
├── api/                     # Routes, dépendances, traduction erreurs -> HTTP
├── core/                    # Configuration (.env) et logging
├── schemas/                 # Modèles Pydantic (entrées, sortie LLM, réponses API)
└── services/
    ├── email_service.py     # Gmail API + mock
    ├── gmail_parser.py      # Message Gmail brut -> EmailInput
    └── llm_service.py       # Prompt, appel Gemini, sortie structurée
scripts/                     # authorize_gmail, try_analysis (essai en ligne de commande)
tests/                       # pytest, sans appel réseau
```

## Qualité et tests

```bash
ruff check . && ruff format --check .   # lint + formatage (`ruff format .` pour corriger)
mypy                                    # typage
pytest                                  # tests, sans appel réseau
```

La CI GitHub Actions (`.github/workflows/ci.yml`) exécute ces mêmes commandes à chaque push
et pull request, avec les tests sur Python 3.11, 3.12 et 3.13. Dependabot propose chaque
semaine les mises à jour des dépendances et des actions.
