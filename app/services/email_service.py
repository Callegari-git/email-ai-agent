"""Service d'extraction des emails : interface commune, implémentation Gmail et mock.

Les appels à l'API Gmail sont bloquants (client HTTP synchrone httplib2) : depuis
FastAPI, ils doivent être exécutés dans un threadpool pour ne pas bloquer la boucle
asyncio.
"""

import logging
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import google_auth_httplib2
import httplib2
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpRequest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas import EmailInput
from app.services.gmail_parser import parse_gmail_message

logger = logging.getLogger(__name__)

# Principe du moindre privilège : lecture seule. L'agent ne peut ni envoyer, ni
# supprimer, ni marquer comme lu. Modifier ce scope impose de régénérer token.json.
GMAIL_SCOPES: list[str] = ["https://www.googleapis.com/auth/gmail.readonly"]

AUTHORIZE_HINT = "Lancez `python -m scripts.authorize_gmail` pour (ré)autoriser l'accès."

_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})


class EmailServiceError(Exception):
    """Erreur de base du service mail."""


class EmailAuthError(EmailServiceError):
    """Identifiants absents, expirés, révoqués ou insuffisants : une action humaine est requise."""


class EmailProviderUnavailableError(EmailServiceError):
    """Erreur transitoire (réseau, timeout, quota, 5xx) : réessayer plus tard peut suffire."""


class EmailNotFoundError(EmailServiceError):
    """Message introuvable (ex: supprimé entre le listing et la récupération)."""


class EmailProvider(Protocol):
    def fetch_unread(self, max_results: int) -> list[EmailInput]:
        """Renvoie les `max_results` emails non lus les plus récents, du plus récent au plus ancien."""
        ...


def _http_error_reasons(exc: HttpError) -> set[str]:
    details = exc.error_details if isinstance(exc.error_details, list) else []
    return {d.get("reason", "") for d in details if isinstance(d, dict)}


def _translate_error(exc: Exception) -> EmailServiceError:
    """Traduit les exceptions des bibliothèques Google en erreurs métier.

    Le reste de l'application (routes) n'a ainsi qu'à distinguer deux cas utiles :
    "il faut ré-autoriser" (EmailAuthError) et "réessayer plus tard"
    (EmailProviderUnavailableError), sans dépendre des types d'exception de Google.
    """
    if isinstance(exc, RefreshError):
        return EmailAuthError(f"Token Gmail expiré ou révoqué. {AUTHORIZE_HINT}")
    if isinstance(exc, HttpError):
        status = exc.status_code
        if status == 404:
            return EmailNotFoundError(f"Ressource Gmail introuvable : {exc.reason}")
        if status == 429 or status >= 500 or (status == 403 and _http_error_reasons(exc) & _RATE_LIMIT_REASONS):
            return EmailProviderUnavailableError(f"API Gmail indisponible ou quota atteint (HTTP {status}).")
        if status in (401, 403):
            # 403 hors quota : scope insuffisant ou API Gmail non activée dans le projet Google Cloud.
            return EmailAuthError(f"Accès Gmail refusé (HTTP {status} : {exc.reason}). {AUTHORIZE_HINT}")
        return EmailServiceError(f"Erreur API Gmail (HTTP {status} : {exc.reason}).")
    # TransportError / HttpLib2Error / OSError (dont TimeoutError) : problème réseau.
    return EmailProviderUnavailableError(f"Impossible de joindre l'API Gmail : {exc}")


class GmailEmailProvider:
    def __init__(
        self,
        token_path: Path,
        query: str = "in:inbox is:unread",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._token_path = token_path
        self._query = query
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    @classmethod
    def from_settings(cls, settings: Settings) -> "GmailEmailProvider":
        return cls(
            token_path=settings.gmail_token_path,
            query=settings.gmail_query,
            timeout_seconds=settings.gmail_timeout_seconds,
            max_retries=settings.gmail_max_retries,
        )

    def fetch_unread(self, max_results: int) -> list[EmailInput]:
        if max_results < 1:
            raise ValueError("max_results doit être >= 1")

        service = self._build_service()
        message_ids = self._list_message_ids(service, max_results)
        logger.info("Gmail : %d email(s) correspondant à la requête %r", len(message_ids), self._query)

        # Un appel par message : acceptable pour quelques dizaines d'emails. Au-delà,
        # les "batch requests" de l'API Gmail réduiraient le nombre d'allers-retours.
        emails = [email for mid in message_ids if (email := self._fetch_message(service, mid)) is not None]
        logger.info("Gmail : %d/%d email(s) récupéré(s) et parsé(s)", len(emails), len(message_ids))
        return emails

    def _build_service(self) -> Any:
        credentials = self._load_credentials()
        # httplib2.Http n'est pas thread-safe : on construit un client par appel à
        # fetch_unread plutôt que de le partager entre requêtes FastAPI concurrentes.
        # Le timeout explicite évite qu'une connexion bloquée ne fige la requête.
        authorized_http = google_auth_httplib2.AuthorizedHttp(
            credentials, http=httplib2.Http(timeout=self._timeout_seconds)
        )
        # La description de l'API Gmail est embarquée dans la bibliothèque : `build` ne fait aucun appel réseau.
        return build("gmail", "v1", http=authorized_http, cache_discovery=False)

    def _load_credentials(self) -> Credentials:
        if not self._token_path.is_file():
            raise EmailAuthError(f"Token Gmail introuvable ({self._token_path}). {AUTHORIZE_HINT}")
        try:
            # Sans passer `scopes` : on veut lire les scopes réellement accordés pour les vérifier.
            credentials = Credentials.from_authorized_user_file(str(self._token_path))
        except (ValueError, OSError) as exc:
            # ValueError couvre aussi un JSON valide mais incomplet (ex: refresh_token absent).
            raise EmailAuthError(f"Token Gmail invalide ou incomplet ({self._token_path}). {AUTHORIZE_HINT}") from exc

        if not credentials.has_scopes(GMAIL_SCOPES):
            raise EmailAuthError(f"Le token Gmail n'a pas les autorisations requises {GMAIL_SCOPES}. {AUTHORIZE_HINT}")

        if credentials.valid:
            return credentials

        # L'access token dure ~1h ; le refresh token permet d'en obtenir un nouveau
        # sans interaction. Piège classique : si l'écran de consentement OAuth du
        # projet Google Cloud est en mode "Testing", le refresh token expire au bout
        # de 7 jours et ce refresh échoue avec une RefreshError.
        logger.info("Access token Gmail expiré, rafraîchissement")
        try:
            credentials.refresh(google_auth_httplib2.Request(httplib2.Http(timeout=self._timeout_seconds)))
        except (RefreshError, TransportError) as exc:
            raise _translate_error(exc) from exc
        self._save_credentials(credentials)
        return credentials

    def _save_credentials(self, credentials: Credentials) -> None:
        """Écriture atomique (fichier temporaire + rename) avec permissions 600.

        Un échec n'est pas bloquant : le token rafraîchi reste valable en mémoire pour
        cet appel, il faudra simplement le rafraîchir à nouveau la prochaine fois.
        """
        try:
            fd, tmp_path = tempfile.mkstemp(dir=self._token_path.parent, prefix=".token-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(credentials.to_json())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._token_path)
        except OSError:
            logger.warning("Impossible de sauvegarder le token Gmail rafraîchi", exc_info=True)

    def _execute(self, request: HttpRequest) -> dict[str, Any]:
        # `num_retries` active le retry intégré au client Google, avec backoff
        # exponentiel, sur les erreurs transitoires (5xx, 429, 403 de quota, erreurs
        # réseau). Les erreurs qui persistent après les retries sont traduites ici.
        try:
            return request.execute(num_retries=self._max_retries)
        except (RefreshError, HttpError, TransportError, httplib2.HttpLib2Error, OSError) as exc:
            raise _translate_error(exc) from exc

    def _list_message_ids(self, service: Any, max_results: int) -> list[str]:
        # max_results est plafonné à 100 par la config, sous la taille de page de
        # l'API (500) : une seule page suffit, pas de pagination à gérer.
        request = service.users().messages().list(userId="me", q=self._query, maxResults=max_results)
        response = self._execute(request)
        return [message["id"] for message in response.get("messages", [])]

    def _fetch_message(self, service: Any, message_id: str) -> EmailInput | None:
        """Un email introuvable ou mal formé est ignoré (log) sans faire échouer le lot.

        Les erreurs d'authentification et de réseau, elles, sont propagées : elles
        toucheraient tous les emails suivants de la même façon.
        """
        request = service.users().messages().get(userId="me", id=message_id, format="full")
        try:
            raw_message = self._execute(request)
        except EmailNotFoundError:
            logger.warning("Email %s introuvable (supprimé entre-temps ?), ignoré", message_id)
            return None
        try:
            return parse_gmail_message(raw_message)
        except (ValidationError, KeyError) as exc:
            logger.warning("Email %s ignoré : format inattendu (%s)", message_id, exc)
            return None


class MockEmailProvider:
    """Boîte mail factice pour développer et tester sans compte Gmail.

    Les emails couvrent volontairement toute l'échelle d'importance, pour pouvoir
    juger le comportement du LLM à l'étape suivante.
    """

    def fetch_unread(self, max_results: int) -> list[EmailInput]:
        if max_results < 1:
            raise ValueError("max_results doit être >= 1")
        emails = _build_mock_emails(now=datetime.now(UTC))[:max_results]
        logger.info("Mock : %d email(s) renvoyé(s)", len(emails))
        return emails


def _build_mock_emails(now: datetime) -> list[EmailInput]:
    return [
        EmailInput(
            id="mock-001",
            sender="Direction Financière <finance@client-important.fr>",
            subject="URGENT - Facture impayée, suspension du service demain",
            body=(
                "Bonjour,\n\nMalgré nos relances, la facture F-2026-0892 de 12 450 € reste impayée. "
                "Sans règlement avant demain 12h, nous serons contraints de suspendre l'accès "
                "à la plateforme pour l'ensemble de vos équipes.\n\n"
                "Merci de nous confirmer la date de virement.\n\nCordialement,\nClaire Dubois"
            ),
            received_at=now - timedelta(minutes=12),
        ),
        EmailInput(
            id="mock-002",
            sender="Marc Lefèvre <marc.lefevre@entreprise.fr>",
            subject="Relecture de la spec API avant jeudi ?",
            body=(
                "Salut,\n\nJ'ai mis à jour la spécification de l'API de facturation. "
                "Tu peux la relire d'ici jeudi ? Surtout la partie sur les webhooks, "
                "j'ai un doute sur la gestion des retries.\n\nMerci !\nMarc"
            ),
            received_at=now - timedelta(hours=2),
        ),
        EmailInput(
            id="mock-003",
            sender="Équipe RH <rh@entreprise.fr>",
            subject="Rappel : nouvelle politique de télétravail",
            body=(
                "Bonjour à tous,\n\nPour rappel, la nouvelle politique de télétravail entre en "
                "vigueur le mois prochain. Le document complet est disponible sur l'intranet. "
                "Aucune action n'est requise de votre part.\n\nL'équipe RH"
            ),
            received_at=now - timedelta(hours=5),
        ),
        EmailInput(
            id="mock-004",
            sender="TechWeekly <newsletter@techweekly.io>",
            subject="Les 10 frameworks Python à suivre cette année",
            body=(
                "Cette semaine : FastAPI continue sa progression, Pydantic v2 domine la "
                "validation de données, et les agents IA envahissent les stacks backend. "
                "Lire la suite sur notre site. Se désinscrire."
            ),
            received_at=now - timedelta(days=1),
        ),
        EmailInput(
            id="mock-005",
            sender="SuperPromo <deals@superpromo.com>",
            subject="-70% sur tout le site, ce week-end seulement !",
            body="Profitez de remises exceptionnelles sur des milliers d'articles. Offre valable jusqu'à dimanche.",
            received_at=now - timedelta(days=2),
        ),
    ]


def get_email_provider(settings: Settings) -> EmailProvider:
    if settings.email_provider == "gmail":
        return GmailEmailProvider.from_settings(settings)
    return MockEmailProvider()
