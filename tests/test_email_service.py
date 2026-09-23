import base64
import json
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httplib2
import pytest
from google.auth.exceptions import RefreshError, TransportError
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

from app.core.config import Settings
from app.services.email_service import (
    GMAIL_SCOPES,
    EmailAuthError,
    EmailNotFoundError,
    EmailProviderUnavailableError,
    EmailServiceError,
    GmailEmailProvider,
    MockEmailProvider,
    _translate_error,
    get_email_provider,
)


def _http_error(status: int, reason: str = "") -> HttpError:
    errors = [{"reason": reason, "message": reason}] if reason else []
    content = json.dumps({"error": {"code": status, "message": reason or "error", "errors": errors}}).encode()
    return HttpError(httplib2.Response({"status": status}), content)


def _raw_message(message_id: str) -> dict[str, Any]:
    data = base64.urlsafe_b64encode(f"Corps de {message_id}".encode()).decode().rstrip("=")
    return {
        "id": message_id,
        "internalDate": "1790000000000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "From", "value": "alice@example.com"}, {"name": "Subject", "value": message_id}],
            "body": {"data": data},
        },
    }


class FakeRequest:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self._result = result
        self.num_retries: int | None = None

    def execute(self, num_retries: int = 0) -> dict[str, Any]:
        self.num_retries = num_retries
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeGmailService:
    """Imite la chaîne `service.users().messages().list/get(...)` du client Google."""

    def __init__(
        self, list_result: dict[str, Any] | Exception, messages: dict[str, dict[str, Any] | Exception]
    ) -> None:
        self.list_result = list_result
        self.messages_by_id = messages
        self.list_kwargs: dict[str, Any] = {}
        self.requests: list[FakeRequest] = []

    def users(self) -> "FakeGmailService":
        return self

    def messages(self) -> "FakeGmailService":
        return self

    def list(self, **kwargs: Any) -> FakeRequest:
        self.list_kwargs = kwargs
        return self._track(FakeRequest(self.list_result))

    def get(self, *, userId: str, id: str, format: str) -> FakeRequest:
        assert (userId, format) == ("me", "full")
        return self._track(FakeRequest(self.messages_by_id[id]))

    def _track(self, request: FakeRequest) -> FakeRequest:
        self.requests.append(request)
        return request


@pytest.fixture
def provider(tmp_path: Path) -> GmailEmailProvider:
    return GmailEmailProvider(token_path=tmp_path / "token.json", query="is:unread", max_retries=2)


def _use_fake_service(monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider, service: FakeGmailService) -> None:
    monkeypatch.setattr(provider, "_build_service", lambda: service)


class TestFetchUnread:
    def test_returns_parsed_emails_in_listing_order(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider
    ) -> None:
        service = FakeGmailService(
            {"messages": [{"id": "m2"}, {"id": "m1"}]}, {"m1": _raw_message("m1"), "m2": _raw_message("m2")}
        )
        _use_fake_service(monkeypatch, provider, service)

        emails = provider.fetch_unread(max_results=5)

        assert [e.id for e in emails] == ["m2", "m1"]
        assert emails[0].body == "Corps de m2"
        assert service.list_kwargs == {"userId": "me", "q": "is:unread", "maxResults": 5}
        assert all(r.num_retries == 2 for r in service.requests)

    def test_empty_inbox(self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider) -> None:
        _use_fake_service(monkeypatch, provider, FakeGmailService({"resultSizeEstimate": 0}, {}))
        assert provider.fetch_unread(max_results=5) == []

    def test_skips_deleted_and_malformed_messages(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider
    ) -> None:
        malformed = _raw_message("m3")
        del malformed["id"]
        service = FakeGmailService(
            {"messages": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]},
            {"m1": _raw_message("m1"), "m2": _http_error(404), "m3": malformed},
        )
        _use_fake_service(monkeypatch, provider, service)

        assert [e.id for e in provider.fetch_unread(max_results=5)] == ["m1"]

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (_http_error(401), EmailAuthError),
            (RefreshError("invalid_grant"), EmailAuthError),
            (_http_error(503), EmailProviderUnavailableError),
            (TimeoutError("timed out"), EmailProviderUnavailableError),
        ],
    )
    def test_listing_errors_are_translated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: GmailEmailProvider,
        error: Exception,
        expected: type[EmailServiceError],
    ) -> None:
        _use_fake_service(monkeypatch, provider, FakeGmailService(error, {}))
        with pytest.raises(expected):
            provider.fetch_unread(max_results=5)

    def test_auth_error_on_a_message_aborts_the_batch(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider
    ) -> None:
        service = FakeGmailService(
            {"messages": [{"id": "m1"}, {"id": "m2"}]}, {"m1": _raw_message("m1"), "m2": _http_error(401)}
        )
        _use_fake_service(monkeypatch, provider, service)
        with pytest.raises(EmailAuthError):
            provider.fetch_unread(max_results=5)

    def test_rejects_invalid_max_results(self, provider: GmailEmailProvider) -> None:
        with pytest.raises(ValueError):
            provider.fetch_unread(max_results=0)


class TestTranslateError:
    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (_http_error(404), EmailNotFoundError),
            (_http_error(401), EmailAuthError),
            (_http_error(403, "insufficientPermissions"), EmailAuthError),
            (_http_error(403, "userRateLimitExceeded"), EmailProviderUnavailableError),
            (_http_error(429), EmailProviderUnavailableError),
            (_http_error(500), EmailProviderUnavailableError),
            (_http_error(400, "badRequest"), EmailServiceError),
            (RefreshError("invalid_grant"), EmailAuthError),
            (TransportError("dns"), EmailProviderUnavailableError),
            (httplib2.ServerNotFoundError("dns"), EmailProviderUnavailableError),
        ],
    )
    def test_maps_to_business_errors(self, error: Exception, expected: type[EmailServiceError]) -> None:
        assert type(_translate_error(error)) is expected


def _write_token(path: Path, *, expired: bool = False, scopes: list[str] = GMAIL_SCOPES, refresh: bool = True) -> None:
    # google-auth manipule des dates UTC "naïves" (sans tzinfo).
    now = datetime.now(UTC).replace(tzinfo=None)
    credentials = Credentials(
        token="access-token",
        refresh_token="refresh-token" if refresh else None,
        token_uri="https://oauth2.googleapis.com/token",
        client_id="client-id",
        client_secret="client-secret",
        scopes=scopes,
        expiry=now - timedelta(hours=1) if expired else now + timedelta(hours=1),
    )
    path.write_text(credentials.to_json(), encoding="utf-8")


class TestLoadCredentials:
    def test_missing_token_file(self, provider: GmailEmailProvider) -> None:
        with pytest.raises(EmailAuthError, match="authorize_gmail"):
            provider._load_credentials()

    def test_corrupted_token_file(self, provider: GmailEmailProvider, tmp_path: Path) -> None:
        (tmp_path / "token.json").write_text("{pas du json", encoding="utf-8")
        with pytest.raises(EmailAuthError):
            provider._load_credentials()

    def test_insufficient_scopes(self, provider: GmailEmailProvider, tmp_path: Path) -> None:
        _write_token(tmp_path / "token.json", scopes=["https://www.googleapis.com/auth/gmail.metadata"])
        with pytest.raises(EmailAuthError, match="autorisations"):
            provider._load_credentials()

    def test_valid_token_is_returned_without_refresh(self, provider: GmailEmailProvider, tmp_path: Path) -> None:
        _write_token(tmp_path / "token.json")
        assert provider._load_credentials().token == "access-token"

    def test_token_without_refresh_token(self, provider: GmailEmailProvider, tmp_path: Path) -> None:
        _write_token(tmp_path / "token.json", refresh=False)
        with pytest.raises(EmailAuthError, match="incomplet"):
            provider._load_credentials()

    def test_revoked_refresh_token(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider, tmp_path: Path
    ) -> None:
        _write_token(tmp_path / "token.json", expired=True)

        def fail_refresh(self: Credentials, request: Any) -> None:
            raise RefreshError("invalid_grant: Token has been expired or revoked.")

        monkeypatch.setattr(Credentials, "refresh", fail_refresh)
        with pytest.raises(EmailAuthError, match="expiré ou révoqué"):
            provider._load_credentials()

    def test_network_error_during_refresh(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider, tmp_path: Path
    ) -> None:
        _write_token(tmp_path / "token.json", expired=True)

        def fail_refresh(self: Credentials, request: Any) -> None:
            raise TransportError("connection reset")

        monkeypatch.setattr(Credentials, "refresh", fail_refresh)
        with pytest.raises(EmailProviderUnavailableError):
            provider._load_credentials()

    def test_refreshed_token_is_persisted_privately(
        self, monkeypatch: pytest.MonkeyPatch, provider: GmailEmailProvider, tmp_path: Path
    ) -> None:
        token_path = tmp_path / "token.json"
        _write_token(token_path, expired=True)

        def fake_refresh(self: Credentials, request: Any) -> None:
            self.token = "new-access-token"
            self.expiry = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)

        monkeypatch.setattr(Credentials, "refresh", fake_refresh)

        assert provider._load_credentials().token == "new-access-token"
        assert json.loads(token_path.read_text())["token"] == "new-access-token"
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        assert list(tmp_path.glob(".token-*")) == []


class TestMockProvider:
    def test_returns_most_recent_first_and_respects_limit(self) -> None:
        emails = MockEmailProvider().fetch_unread(max_results=3)
        assert len(emails) == 3
        dates = [e.received_at for e in emails if e.received_at is not None]
        assert len(dates) == 3
        assert dates == sorted(dates, reverse=True)

    def test_limit_above_available(self) -> None:
        emails = MockEmailProvider().fetch_unread(max_results=100)
        assert len({e.id for e in emails}) == len(emails) == 5

    def test_rejects_invalid_max_results(self) -> None:
        with pytest.raises(ValueError):
            MockEmailProvider().fetch_unread(max_results=0)


class TestGetEmailProvider:
    def test_mock_by_default(self) -> None:
        assert isinstance(get_email_provider(Settings()), MockEmailProvider)

    def test_gmail_from_settings(self, tmp_path: Path) -> None:
        settings = Settings(email_provider="gmail", gmail_token_path=tmp_path / "t.json", gmail_query="is:starred")
        provider = get_email_provider(settings)
        assert isinstance(provider, GmailEmailProvider)
        assert provider._query == "is:starred"
