from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from app.core import config
from app.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Empêche le vrai `.env` du développeur d'influencer les tests.
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_without_environment() -> None:
    settings = get_settings()
    assert settings.email_provider == "mock"
    assert settings.gemini_api_key is None
    assert settings.max_emails == 10


def test_reads_and_coerces_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_EMAILS", "25")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = get_settings()

    assert settings.max_emails == 25
    assert settings.log_level == "DEBUG"
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)


def test_empty_variable_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert get_settings().gemini_api_key is None


@pytest.mark.parametrize(("name", "value"), [("MAX_EMAILS", "0"), ("EMAIL_PROVIDER", "pop3"), ("GMAIL_TIMEOUT_SECONDS", "abc")])
def test_invalid_values_fail_fast(monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        get_settings()
