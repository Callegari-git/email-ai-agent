"""Dépendances FastAPI : accès aux services créés au démarrage (voir `lifespan` dans main.py).

Passer par `Depends` plutôt que par des variables globales permet aux tests de
substituer les services via `app.dependency_overrides`.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.email_service import EmailProvider
from app.services.llm_service import EmailAnalyzer


def provide_email_provider(request: Request) -> EmailProvider:
    return request.app.state.email_provider


def provide_analyzer(request: Request) -> EmailAnalyzer:
    return request.app.state.analyzer


SettingsDep = Annotated[Settings, Depends(get_settings)]
EmailProviderDep = Annotated[EmailProvider, Depends(provide_email_provider)]
AnalyzerDep = Annotated[EmailAnalyzer, Depends(provide_analyzer)]
