from app.schemas.api import ErrorResponse, HealthResponse
from app.schemas.email import (
    AnalysisFailure,
    AnalyzedEmail,
    EmailAnalysisResult,
    EmailBatchAnalysisResponse,
    EmailInput,
    ImportanceLevel,
)

__all__ = [
    "AnalysisFailure",
    "AnalyzedEmail",
    "EmailAnalysisResult",
    "EmailBatchAnalysisResponse",
    "EmailInput",
    "ErrorResponse",
    "HealthResponse",
    "ImportanceLevel",
]
