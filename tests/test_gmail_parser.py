import base64
from datetime import UTC, datetime
from typing import Any

import pytest

from app.services.gmail_parser import extract_body, html_to_text, parse_gmail_message


def _b64(text: str, encoding: str = "utf-8") -> str:
    # Reproduit l'encodage de Gmail : base64 URL-safe sans padding.
    return base64.urlsafe_b64encode(text.encode(encoding)).decode("ascii").rstrip("=")


def _part(mime_type: str, text: str, *, filename: str = "", charset: str | None = None) -> dict[str, Any]:
    headers = [{"name": "Content-Type", "value": f"{mime_type}; charset={charset}"}] if charset else []
    encoding = charset or "utf-8"
    return {"mimeType": mime_type, "filename": filename, "headers": headers, "body": {"data": _b64(text, encoding)}}


def _message(payload: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    payload.setdefault("headers", [])
    payload["headers"] += [
        {"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "Réunion demain"},
    ]
    return {"id": "abc123", "internalDate": "1790000000000", "snippet": "Aperçu", "payload": payload} | overrides


class TestExtractBody:
    def test_single_plain_text_part(self) -> None:
        assert extract_body(_part("text/plain", "Bonjour,\r\n\r\nÀ demain.")) == "Bonjour,\n\nÀ demain."

    def test_prefers_plain_text_in_multipart_alternative(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [_part("text/plain", "Version texte"), _part("text/html", "<p>Version HTML</p>")],
        }
        assert extract_body(payload) == "Version texte"

    def test_falls_back_to_html_when_plain_is_empty(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [_part("text/plain", "   "), _part("text/html", "<p>Contenu <b>HTML</b></p>")],
        }
        assert extract_body(payload) == "Contenu HTML"

    def test_walks_nested_parts_and_ignores_attachments(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "multipart/alternative", "parts": [_part("text/plain", "Corps principal")]},
                _part("text/plain", "Contenu du fichier joint", filename="notes.txt"),
            ],
        }
        assert extract_body(payload) == "Corps principal"

    def test_decodes_declared_charset(self) -> None:
        assert extract_body(_part("text/plain", "Échéance reçue", charset="iso-8859-1")) == "Échéance reçue"

    def test_unknown_charset_falls_back_to_utf8(self) -> None:
        part = _part("text/plain", "Texte")
        part["headers"] = [{"name": "Content-Type", "value": "text/plain; charset=x-inexistant"}]
        assert extract_body(part) == "Texte"

    def test_invalid_base64_is_ignored(self) -> None:
        payload = {"mimeType": "text/plain", "body": {"data": "@@@"}}
        assert extract_body(payload) == ""

    def test_part_without_data_returns_empty(self) -> None:
        assert extract_body({"mimeType": "text/plain", "body": {"size": 0}}) == ""


class TestHtmlToText:
    def test_strips_tags_scripts_and_styles(self) -> None:
        html = (
            "<html><head><title>T</title><style>p{color:red}</style></head>"
            "<body><script>alert(1)</script><h1>Titre</h1><p>Paragraphe&nbsp;1 &amp; 2</p>"
            "Ligne<br>suivante<ul><li>A</li><li>B</li></ul></body></html>"
        )
        assert html_to_text(html) == "Titre\n\nParagraphe 1 & 2\nLigne\nsuivante\nA\n\nB"


class TestParseGmailMessage:
    def test_builds_email_input(self) -> None:
        email = parse_gmail_message(_message(_part("text/plain", "Corps")))
        assert email.id == "abc123"
        assert email.sender == "Alice <alice@example.com>"
        assert email.subject == "Réunion demain"
        assert email.body == "Corps"
        assert email.received_at == datetime.fromtimestamp(1_790_000_000, tz=UTC)

    def test_headers_are_case_insensitive(self) -> None:
        message = _message({"mimeType": "text/plain", "body": {}})
        message["payload"]["headers"] = [{"name": "FROM", "value": "bob@example.com"}]
        assert parse_gmail_message(message).sender == "bob@example.com"

    def test_missing_headers_use_placeholders(self) -> None:
        message = _message({"mimeType": "text/plain", "body": {}})
        message["payload"]["headers"] = []
        email = parse_gmail_message(message)
        assert email.sender == "(expéditeur inconnu)"
        assert email.subject == ""

    def test_empty_body_falls_back_to_snippet(self) -> None:
        assert parse_gmail_message(_message({"mimeType": "text/plain", "body": {}})).body == "Aperçu"

    @pytest.mark.parametrize("internal_date", [None, "pas-un-nombre"])
    def test_invalid_internal_date_is_none(self, internal_date: str | None) -> None:
        message = _message(_part("text/plain", "Corps"), internalDate=internal_date)
        assert parse_gmail_message(message).received_at is None

    def test_missing_id_raises(self) -> None:
        message = _message(_part("text/plain", "Corps"))
        del message["id"]
        with pytest.raises(KeyError):
            parse_gmail_message(message)
