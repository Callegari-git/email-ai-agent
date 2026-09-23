"""Doubles de test partagés : faux client Gemini (aucun appel réseau)."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

from google.genai import types

VALID_ANALYSIS = {
    "importance_reason": "Le client exige un paiement avant demain midi.",
    "importance": 5,
    "summary": ["Facture de 12 450 € impayée.", "Suspension du service demain 12h sans virement."],
}


def make_response(
    text: str | None = None, *, finish_reason: types.FinishReason = types.FinishReason.STOP
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text or json.dumps(VALID_ANALYSIS))]),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=420, candidates_token_count=80),
    )


Handler = Callable[[str], types.GenerateContentResponse]


class FakeModels:
    """Imite `client.aio.models` : enregistre les appels et délègue la réponse à un handler."""

    def __init__(self, handler: Handler, delay: float = 0.0) -> None:
        self._handler = handler
        self._delay = delay
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate_content(self, *, model: str, contents: str, config: types.GenerateContentConfig) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay)
            return self._handler(contents)
        finally:
            self.in_flight -= 1


class FakeAio:
    def __init__(self, models: FakeModels) -> None:
        self.models = models
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, handler: Handler, delay: float = 0.0) -> None:
        self.aio = FakeAio(FakeModels(handler, delay))
