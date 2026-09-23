"""Essai manuel du pipeline complet : `python -m scripts.try_analysis [nombre_emails]`.

Récupère les emails via le fournisseur configuré (EMAIL_PROVIDER), les analyse avec
Gemini et affiche le JSON obtenu. Utile pour ajuster le prompt avant de passer par l'API.
"""

import asyncio
import logging
import sys

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.services.email_service import EmailServiceError, get_email_provider
from app.services.llm_service import EmailAnalyzer, LLMServiceError

logger = logging.getLogger("try_analysis")


async def run(max_emails: int) -> int:
    settings = get_settings()
    try:
        analyzer = EmailAnalyzer.from_settings(settings)
        emails = get_email_provider(settings).fetch_unread(max_emails)
    except (EmailServiceError, LLMServiceError) as exc:
        logger.error("%s", exc)
        return 1

    try:
        result = await analyzer.analyze_emails(emails)
    except LLMServiceError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        await analyzer.aclose()

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    setup_logging(get_settings().log_level)
    count = int(sys.argv[1]) if len(sys.argv) > 1 else get_settings().max_emails
    sys.exit(asyncio.run(run(count)))
