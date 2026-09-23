"""Point d'entrée de l'API : `uvicorn app.main:app --reload`, puis http://127.0.0.1:8000/docs."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_error_handlers
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.services.email_service import get_email_provider
from app.services.llm_service import EmailAnalyzer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Crée les services une fois au démarrage et libère leurs ressources à l'arrêt.

    Le client Gemini garde un pool de connexions HTTP : le partager entre les requêtes
    évite de renégocier une connexion TLS à chaque analyse. Une configuration invalide
    (ex: clé API absente) fait échouer le démarrage plutôt que la première requête.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    app.state.email_provider = get_email_provider(settings)
    app.state.analyzer = EmailAnalyzer.from_settings(settings)
    logger.info("Démarrage : fournisseur mail=%s, modèle=%s", settings.email_provider, settings.gemini_model)
    try:
        yield
    finally:
        await app.state.analyzer.aclose()
        logger.info("Arrêt : ressources libérées")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Email Triage Agent",
        version="0.1.0",
        summary="Tri et résumé d'emails par IA (Gmail + Gemini, sorties structurées).",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
