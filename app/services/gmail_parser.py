"""Conversion d'un message brut de l'API Gmail (format="full") vers `EmailInput`.

Module sans I/O : il ne fait que transformer des dicts, ce qui le rend testable
sans compte Gmail. Référence du format :
https://developers.google.com/gmail/api/reference/rest/v1/users.messages
"""

import base64
import binascii
import logging
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from email.message import Message
from html.parser import HTMLParser
from typing import Any

from app.schemas import EmailInput

logger = logging.getLogger(__name__)

GmailMessage = dict[str, Any]
GmailPart = dict[str, Any]

_BLOCK_TAGS = frozenset({"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "blockquote"})
_SKIPPED_TAGS = frozenset({"script", "style", "head", "title"})


class _HTMLToTextParser(HTMLParser):
    """Extraction de texte minimaliste, suffisante pour un LLM : on garde le contenu
    et les sauts de ligne structurants, sans dépendre de BeautifulSoup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _HTMLToTextParser()
    parser.feed(html)
    parser.close()
    return normalize_whitespace(parser.get_text())


def normalize_whitespace(text: str) -> str:
    """Espaces multiples réduits, lignes nettoyées, au plus une ligne vide consécutive."""
    lines = (re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def get_header(message: GmailMessage, name: str) -> str:
    """Les noms d'en-têtes sont insensibles à la casse (RFC 5322)."""
    headers: list[dict[str, str]] = message.get("payload", {}).get("headers", [])
    target = name.lower()
    return next((h.get("value", "") for h in headers if h.get("name", "").lower() == target), "")


def _part_charset(part: GmailPart) -> str:
    content_type = next(
        (h.get("value", "") for h in part.get("headers", []) if h.get("name", "").lower() == "content-type"),
        "",
    )
    if not content_type:
        return "utf-8"
    parsed = Message()
    parsed["Content-Type"] = content_type
    return parsed.get_content_charset() or "utf-8"


def _decode_part_body(part: GmailPart) -> str:
    data: str | None = part.get("body", {}).get("data")
    if not data:
        return ""
    # Gmail encode en base64 "URL-safe" et retire parfois le padding '=' final.
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (binascii.Error, ValueError):
        logger.warning("Partie MIME illisible (base64 invalide), ignorée")
        return ""
    charset = _part_charset(part)
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _iter_parts(part: GmailPart) -> Iterator[GmailPart]:
    yield part
    for child in part.get("parts", []) or []:
        yield from _iter_parts(child)


def extract_body(payload: GmailPart) -> str:
    """Parcourt l'arbre MIME et renvoie le corps en texte brut.

    Stratégie : un email "multipart/alternative" contient souvent la même chose en
    text/plain et en text/html. On préfère text/plain (plus fidèle, moins de bruit
    de mise en page) et on ne se rabat sur le HTML converti que s'il est absent ou
    vide, ce qui est fréquent pour les newsletters. Les pièces jointes (parties
    avec un `filename`) sont ignorées, même lorsqu'elles sont de type texte.
    """
    plain_chunks: list[str] = []
    html_chunks: list[str] = []
    for part in _iter_parts(payload):
        if part.get("filename"):
            continue
        mime_type = part.get("mimeType", "").lower()
        if mime_type == "text/plain":
            plain_chunks.append(_decode_part_body(part))
        elif mime_type == "text/html":
            html_chunks.append(_decode_part_body(part))

    plain = normalize_whitespace("\n\n".join(plain_chunks))
    if plain:
        return plain
    return html_to_text("\n".join(html_chunks))


def _parse_internal_date(message: GmailMessage) -> datetime | None:
    # `internalDate` (ms depuis epoch, date de réception par Gmail) est plus fiable
    # que l'en-tête `Date`, qui est déclaré librement par l'expéditeur.
    raw = message.get("internalDate")
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc) if raw else None
    except (TypeError, ValueError, OverflowError):
        return None


def parse_gmail_message(message: GmailMessage) -> EmailInput:
    payload: GmailPart = message.get("payload", {})
    body = extract_body(payload) or normalize_whitespace(message.get("snippet", ""))
    return EmailInput(
        id=message["id"],
        sender=get_header(message, "From") or "(expéditeur inconnu)",
        subject=get_header(message, "Subject"),
        body=body,
        received_at=_parse_internal_date(message),
    )
