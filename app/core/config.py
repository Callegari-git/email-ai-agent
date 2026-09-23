import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseModel):
    """Configuration de l'application, lue depuis les variables d'environnement.

    Chaque champ correspond à la variable d'environnement du même nom en majuscules
    (ex: `gemini_model` <- `GEMINI_MODEL`). Pydantic se charge du typage et de la
    validation : une valeur invalide fait échouer le démarrage plutôt qu'une requête.
    """

    model_config = ConfigDict(frozen=True)

    # SecretStr masque la valeur dans les repr/logs (affiche '**********').
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    # None = niveau par défaut du modèle. Les valeurs acceptées dépendent du modèle
    # (ex: gemini-3.8-flash n'accepte pas "minimal") : ne le fixer qu'en connaissance de cause.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_max_body_chars: int = Field(default=8000, ge=500)
    llm_max_concurrency: int = Field(default=5, ge=1, le=50)

    email_provider: Literal["mock", "gmail"] = "mock"
    gmail_credentials_path: Path = Path("credentials.json")
    gmail_token_path: Path = Path("token.json")
    gmail_query: str = "in:inbox is:unread"
    gmail_timeout_seconds: float = Field(default=30.0, gt=0)
    gmail_max_retries: int = Field(default=3, ge=0, le=10)
    max_emails: int = Field(default=10, ge=1, le=100)

    log_level: LogLevel = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """Charge la configuration une seule fois (singleton via lru_cache)."""
    load_dotenv()
    # Les variables vides (ex: `IMAP_PASSWORD=`) sont ignorées pour retomber sur la valeur par défaut.
    raw: dict[str, str] = {
        field: value for field in Settings.model_fields if (value := os.environ.get(field.upper(), "").strip())
    }
    return Settings.model_validate(raw)
