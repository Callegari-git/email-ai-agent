from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalyzedEmail,
    EmailAnalysisResult,
    EmailInput,
    ImportanceLevel,
)


def _valid_analysis_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "importance_reason": "Le client demande une validation du devis avant vendredi.",
        "importance": 4,
        "summary": ["Le client attend la validation du devis.", "Échéance : vendredi."],
    }
    return payload | overrides


class TestEmailInput:
    def test_strips_whitespace_and_applies_defaults(self) -> None:
        email = EmailInput(id="  42 ", sender=" alice@example.com ")
        assert email.id == "42"
        assert email.sender == "alice@example.com"
        assert email.subject == ""
        assert email.body == ""
        assert email.received_at is None

    @pytest.mark.parametrize("field", ["id", "sender"])
    def test_rejects_blank_required_fields(self, field: str) -> None:
        data = {"id": "42", "sender": "alice@example.com", field: "   "}
        with pytest.raises(ValidationError):
            EmailInput.model_validate(data)

    def test_is_immutable(self) -> None:
        email = EmailInput(id="42", sender="alice@example.com")
        with pytest.raises(ValidationError):
            email.subject = "modifié"  # type: ignore[misc]


class TestEmailAnalysisResult:
    def test_parses_llm_json(self) -> None:
        result = EmailAnalysisResult.model_validate_json(
            '{"importance_reason": "Promo.", "importance": 1, "summary": ["Newsletter commerciale."]}'
        )
        assert result.importance is ImportanceLevel.NEGLIGIBLE

    @pytest.mark.parametrize("importance", [0, 6, "haute"])
    def test_rejects_out_of_range_importance(self, importance: Any) -> None:
        with pytest.raises(ValidationError):
            EmailAnalysisResult(**_valid_analysis_payload(importance=importance))

    @pytest.mark.parametrize("summary", [[], ["a", "b", "c", "d"], ["  ", ""]])
    def test_rejects_invalid_summary(self, summary: list[str]) -> None:
        with pytest.raises(ValidationError):
            EmailAnalysisResult(**_valid_analysis_payload(summary=summary))

    def test_cleans_summary_lines(self) -> None:
        result = EmailAnalysisResult(**_valid_analysis_payload(summary=["  Ligne 1 ", "", "Ligne 2"]))
        assert result.summary == ["Ligne 1", "Ligne 2"]

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            EmailAnalysisResult(**_valid_analysis_payload(extra_field="x"))

    def test_json_schema_is_compatible_with_gemini(self) -> None:
        # Vérifie hors ligne ce que l'API Gemini rejetterait sinon à l'exécution.
        schema = EmailAnalysisResult.model_json_schema()
        properties = schema["properties"]

        _assert_gemini_compatible(schema)
        assert set(schema["required"]) == set(properties)
        # L'ordre porte le raisonnement "justification avant note" (voir le schéma).
        assert list(properties) == ["importance_reason", "importance", "summary"]
        assert properties["importance"] == {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": properties["importance"]["description"],
        }
        assert properties["summary"]["maxItems"] == 3
        assert all("description" in prop for prop in properties.values())


# Mots-clés JSON Schema acceptés par `response_json_schema` (doc du SDK google-genai).
_GEMINI_SCHEMA_KEYWORDS = frozenset(
    {
        "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum",
        "items", "prefixItems", "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf",
        "properties", "additionalProperties", "required", "propertyOrdering",
    }
)  # fmt: skip


def _assert_gemini_compatible(node: dict[str, Any]) -> None:
    unsupported = set(node) - _GEMINI_SCHEMA_KEYWORDS
    assert not unsupported, f"Mots-clés non supportés par Gemini : {unsupported}"
    if "$ref" in node:
        assert all(key.startswith("$") for key in node), f"$ref avec propriétés voisines : {node}"
    for key in ("properties", "$defs"):
        for child in node.get(key, {}).values():
            _assert_gemini_compatible(child)
    for key in ("items", "additionalProperties"):
        if isinstance(node.get(key), dict):
            _assert_gemini_compatible(node[key])
    for child in node.get("anyOf", []) + node.get("oneOf", []) + node.get("prefixItems", []):
        _assert_gemini_compatible(child)


class TestAnalyzedEmail:
    def test_from_email_copies_metadata_from_source(self) -> None:
        received_at = datetime(2026, 9, 22, 9, 30, tzinfo=UTC)
        email = EmailInput(id="42", sender="alice@example.com", subject="Devis", received_at=received_at)
        analysis = EmailAnalysisResult(**_valid_analysis_payload())

        analyzed = AnalyzedEmail.from_email(email, analysis)

        assert analyzed.email_id == "42"
        assert analyzed.subject == "Devis"
        assert analyzed.received_at == received_at
        assert analyzed.analysis == analysis
