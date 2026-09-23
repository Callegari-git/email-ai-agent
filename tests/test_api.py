import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import SecretStr

from app import main
from app.api.dependencies import provide_analyzer, provide_email_provider
from app.core.config import Settings, get_settings
from app.schemas import EmailInput
from app.services.email_service import (
    EmailAuthError,
    EmailProviderUnavailableError,
    EmailServiceError,
    MockEmailProvider,
)
from app.services.llm_service import EmailAnalyzer, LLMConfigurationError
from tests.fakes import VALID_ANALYSIS, FakeClient, make_response

# Note attribuée par le faux Gemini selon l'objet des emails du MockEmailProvider.
_IMPORTANCE_BY_SUBJECT = {"Facture impayée": 5, "spec API": 3, "frameworks Python": 1, "-70%": 1}


def _fake_gemini(contents: str) -> types.GenerateContentResponse:
    if "télétravail" in contents:
        raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    importance = next((score for key, score in _IMPORTANCE_BY_SUBJECT.items() if key in contents), 2)
    return make_response(json.dumps(VALID_ANALYSIS | {"importance": importance}))


class FailingProvider:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def fetch_unread(self, max_results: int) -> list[EmailInput]:
        raise self._error


@pytest.fixture
def app() -> FastAPI:
    application = main.create_app()
    analyzer = EmailAnalyzer(FakeClient(_fake_gemini), "gemini-test")  # type: ignore[arg-type]
    application.dependency_overrides[provide_email_provider] = MockEmailProvider
    application.dependency_overrides[provide_analyzer] = lambda: analyzer
    # Isole les tests du `.env` local du développeur.
    application.dependency_overrides[get_settings] = lambda: Settings(gemini_model="gemini-test")
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    # Sans bloc `with`, TestClient n'exécute pas le lifespan : les services viennent
    # des dependency_overrides, aucun vrai client Gmail/Gemini n'est créé.
    return TestClient(app)


def _override_provider(app: FastAPI, error: Exception) -> None:
    app.dependency_overrides[provide_email_provider] = lambda: FailingProvider(error)


class TestSystemRoutes:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "email_provider": "mock", "llm_model": "gemini-test"}

    def test_root_redirects_to_swagger(self, client: TestClient) -> None:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"

    def test_openapi_exposes_examples_and_error_schema(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        analyze = spec["paths"]["/emails/analyze"]["post"]
        examples = analyze["requestBody"]["content"]["application/json"]["examples"]
        assert {"facture_urgente", "newsletter", "injection_de_prompt"} <= set(examples)
        assert analyze["responses"]["503"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


class TestListUnreadEmails:
    def test_returns_emails_with_default_limit(self, client: TestClient) -> None:
        response = client.get("/emails/unread")
        assert response.status_code == 200
        assert len(response.json()) == 5

    def test_respects_max_results(self, client: TestClient) -> None:
        emails = client.get("/emails/unread", params={"max_results": 2}).json()
        assert [e["id"] for e in emails] == ["mock-001", "mock-002"]

    @pytest.mark.parametrize("max_results", [0, 101, "abc"])
    def test_rejects_invalid_max_results(self, client: TestClient, max_results: Any) -> None:
        assert client.get("/emails/unread", params={"max_results": max_results}).status_code == 422


class TestAnalyzeUnreadEmails:
    def test_sorted_by_priority_with_failures_listed(self, client: TestClient) -> None:
        response = client.post("/emails/unread/analyze")

        assert response.status_code == 200
        body = response.json()
        importances = [item["analysis"]["importance"] for item in body["analyzed"]]
        assert importances == [5, 3, 1, 1]
        # À importance égale, le plus récent d'abord (newsletter d'hier avant la promo d'avant-hier).
        assert [item["email_id"] for item in body["analyzed"]][2:] == ["mock-004", "mock-005"]
        assert body["failed"] == [
            {"email_id": "mock-003", "error": "API Gemini indisponible ou quota atteint (HTTP 503)."}
        ]

    def test_without_sorting_keeps_mailbox_order(self, client: TestClient) -> None:
        body = client.post("/emails/unread/analyze", params={"sort_by_priority": False}).json()
        assert [item["email_id"] for item in body["analyzed"]] == ["mock-001", "mock-002", "mock-004", "mock-005"]

    def test_response_contract(self, client: TestClient) -> None:
        item = client.post("/emails/unread/analyze", params={"max_results": 1}).json()["analyzed"][0]
        assert set(item) == {"email_id", "sender", "subject", "received_at", "analysis"}
        assert set(item["analysis"]) == {"importance_reason", "importance", "summary"}

    def test_llm_configuration_error_returns_500(self, app: FastAPI, client: TestClient) -> None:
        def invalid_key(contents: str) -> types.GenerateContentResponse:
            payload = {"error": {"code": 400, "message": "API key not valid", "details": [{"reason": "API_KEY_INVALID"}]}}
            raise genai_errors.ClientError(400, payload)

        analyzer = EmailAnalyzer(FakeClient(invalid_key), "gemini-test")  # type: ignore[arg-type]
        app.dependency_overrides[provide_analyzer] = lambda: analyzer

        response = client.post("/emails/unread/analyze")

        assert response.status_code == 500
        assert response.json()["error"] == "llm_configuration_error"


class TestAnalyzeEmail:
    def test_analyzes_submitted_email(self, client: TestClient) -> None:
        payload = {"id": "x1", "sender": "a@b.c", "subject": "Facture impayée", "body": "Payer avant demain."}

        response = client.post("/emails/analyze", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["email_id"] == "x1"
        assert body["analysis"]["importance"] == 5
        assert body["received_at"] is None

    def test_invalid_body_returns_422(self, client: TestClient) -> None:
        assert client.post("/emails/analyze", json={"id": "", "sender": "a@b.c"}).status_code == 422

    def test_llm_unavailable_returns_503_with_retry_after(self, client: TestClient) -> None:
        response = client.post("/emails/analyze", json={"id": "x1", "sender": "a@b.c", "subject": "Politique de télétravail"})

        assert response.status_code == 503
        assert response.headers["retry-after"] == "30"
        assert response.json()["error"] == "llm_unavailable"


class TestEmailErrorMapping:
    @pytest.mark.parametrize(
        ("error", "status", "code", "retry_after"),
        [
            (EmailAuthError("Token Gmail expiré."), 503, "email_auth_required", False),
            (EmailProviderUnavailableError("Gmail injoignable."), 503, "email_provider_unavailable", True),
            (EmailServiceError("Erreur API Gmail (HTTP 400)."), 502, "email_provider_error", False),
        ],
    )
    def test_email_errors(
        self, app: FastAPI, client: TestClient, error: Exception, status: int, code: str, retry_after: bool
    ) -> None:
        _override_provider(app, error)

        for method, path in (("GET", "/emails/unread"), ("POST", "/emails/unread/analyze")):
            response = client.request(method, path)
            assert response.status_code == status
            assert response.json() == {"error": code, "detail": str(error)}
            assert ("retry-after" in response.headers) is retry_after


class TestLifespan:
    @pytest.fixture(autouse=True)
    def quiet_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Évite que le lifespan reconfigure le logging global pendant la suite de tests.
        monkeypatch.setattr(main, "setup_logging", lambda level: None)

    def test_creates_and_closes_services(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = Settings(gemini_api_key=SecretStr("clé-test"))
        monkeypatch.setattr(main, "get_settings", lambda: settings)
        closed: list[bool] = []

        async def record_close(self: EmailAnalyzer) -> None:
            closed.append(True)

        monkeypatch.setattr(EmailAnalyzer, "aclose", record_close)
        app = main.create_app()
        app.dependency_overrides[get_settings] = lambda: settings

        with TestClient(app) as client:
            assert isinstance(app.state.analyzer, EmailAnalyzer)
            assert isinstance(app.state.email_provider, MockEmailProvider)
            assert client.get("/health").status_code == 200
            assert closed == []
        assert closed == [True]

    def test_startup_fails_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main, "get_settings", lambda: Settings())
        with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
            with TestClient(main.create_app()):
                pass
