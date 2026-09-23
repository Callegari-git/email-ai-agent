from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Format commun des erreurs renvoyées par l'API."""

    error: str = Field(description="Code d'erreur stable, exploitable par un client.", examples=["llm_unavailable"])
    detail: str = Field(description="Message lisible expliquant l'erreur et, si possible, comment la résoudre.")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    email_provider: str
    llm_model: str
