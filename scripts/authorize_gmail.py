"""Autorisation OAuth Gmail, à lancer une fois en local : `python -m scripts.authorize_gmail`.

Ouvre le navigateur pour le consentement Google, puis écrit le token (access +
refresh token) dans GMAIL_TOKEN_PATH. L'API FastAPI ne fait ensuite que relire et
rafraîchir ce token : elle ne déclenche jamais de flux interactif, impossible à
mener depuis une requête HTTP.
"""

import logging
import os
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.services.email_service import GMAIL_SCOPES

logger = logging.getLogger("authorize_gmail")


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    if not settings.gmail_credentials_path.is_file():
        logger.error(
            "Fichier client OAuth introuvable : %s. Créez un identifiant OAuth de type "
            "'Application de bureau' dans la Google Cloud Console (API Gmail activée) "
            "et téléchargez-le à cet emplacement.",
            settings.gmail_credentials_path,
        )
        return 1

    flow = InstalledAppFlow.from_client_secrets_file(str(settings.gmail_credentials_path), GMAIL_SCOPES)
    # prompt="consent" force Google à renvoyer un refresh token même si l'application
    # a déjà été autorisée (sinon il n'est fourni qu'au tout premier consentement).
    credentials = flow.run_local_server(port=0, prompt="consent")

    token_path = settings.gmail_token_path
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    os.chmod(token_path, 0o600)
    logger.info("Token Gmail enregistré dans %s", token_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
