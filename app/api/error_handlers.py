"""Traduction des erreurs métier des services en réponses HTTP.

Les services ne connaissent pas HTTP, et les routes n'ont pas de try/except : chaque
exception métier non gérée remonte jusqu'ici et devient une réponse `ErrorResponse`.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import ErrorResponse
from app.services.email_service import EmailAuthError, EmailProviderUnavailableError, EmailServiceError
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseError,
    LLMServiceError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

RETRY_AFTER_SECONDS = 30

# Choix des codes : le client de cette API n'est pas en cause dans ces erreurs, d'où
# aucun 4xx. En particulier, un token Gmail expiré n'est pas un 401 : ce code dirait
# au client de s'authentifier auprès de NOTRE API, alors que c'est le serveur qui doit
# renouveler son accès à Gmail. 503 = service inutilisable pour le moment ;
# 502 = réponse invalide d'un service externe ; 500 = serveur mal configuré.
_ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    EmailAuthError: (503, "email_auth_required"),
    EmailProviderUnavailableError: (503, "email_provider_unavailable"),
    EmailServiceError: (502, "email_provider_error"),
    LLMConfigurationError: (500, "llm_configuration_error"),
    LLMUnavailableError: (503, "llm_unavailable"),
    LLMResponseError: (502, "llm_invalid_response"),
    LLMServiceError: (502, "llm_error"),
}

# Seules les erreurs transitoires invitent le client à réessayer.
_RETRYABLE = (EmailProviderUnavailableError, LLMUnavailableError)


async def handle_service_error(request: Request, exc: Exception) -> JSONResponse:
    # Parcours du MRO : l'exception la plus spécifique l'emporte (EmailAuthError avant
    # EmailServiceError), quel que soit l'ordre de déclaration dans _ERROR_MAP.
    status_code, code = next(_ERROR_MAP[cls] for cls in type(exc).__mro__ if cls in _ERROR_MAP)

    log_level = logging.ERROR if status_code == 500 else logging.WARNING
    logger.log(log_level, "%s %s -> %d %s : %s", request.method, request.url.path, status_code, code, exc)

    headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if isinstance(exc, _RETRYABLE) else None
    body = ErrorResponse(error=code, detail=str(exc))
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    for exception_class in _ERROR_MAP:
        app.add_exception_handler(exception_class, handle_service_error)
