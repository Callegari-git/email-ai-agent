from datetime import datetime
from enum import IntEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema, field_validator


class ImportanceLevel(IntEnum):
    NEGLIGIBLE = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


class EmailInput(BaseModel):
    """Email normalisé, indépendant de sa source (IMAP, API Gmail, mock).

    Le service mail convertit ses données brutes vers ce modèle : le reste de
    l'application (LLM, routes) ne connaît jamais le format du fournisseur.
    """

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1, description="Identifiant unique côté fournisseur (UID IMAP, id Gmail).")
    sender: str = Field(min_length=1, examples=["Alice Martin <alice@example.com>"])
    subject: str = ""
    body: str = Field(default="", description="Corps de l'email en texte brut (HTML déjà converti).")
    received_at: datetime | None = None


# Gemini n'accepte aucune propriété à côté d'un `$ref` : or Pydantic représente un
# Enum par une référence vers `$defs`, à laquelle il ajoute la `description` du champ.
# On garde l'IntEnum côté Python (valeurs nommées) mais on génère un schéma inline.
ImportanceField = Annotated[
    ImportanceLevel,
    WithJsonSchema({"type": "integer", "enum": [level.value for level in ImportanceLevel]}),
]


# Schéma de sortie imposé au LLM via les Structured Outputs de Gemini.
#
# Ce modèle est converti en JSON Schema et envoyé tel quel à l'API : la docstring
# de la classe et les `description` des champs font donc partie du prompt. Elles
# sont rédigées pour le modèle ; les notes pour développeurs restent en commentaires.
#
# Gemini ne supporte qu'un sous-ensemble de JSON Schema (type, enum, minItems,
# maxItems, minimum, maximum, required...). Tout contrôle hors de ce sous-ensemble
# (ex: longueur d'une chaîne) passe par un validateur Pydantic, exécuté après la
# génération sans apparaître dans le schéma. Tous les champs restent obligatoires.
class EmailAnalysisResult(BaseModel):
    """Analyse d'un email : niveau d'importance justifié et résumé court."""

    model_config = ConfigDict(extra="forbid")

    # L'ordre des champs est volontaire. Le LLM génère le JSON dans l'ordre du
    # schéma, token par token : en lui faisant écrire sa justification AVANT la
    # note, la note est conditionnée par ce raisonnement (mini chain-of-thought),
    # ce qui donne des scores plus cohérents qu'une note produite "à froid".
    importance_reason: str = Field(
        description=(
            "Une phrase courte justifiant le niveau d'importance : qui écrit, "
            "ce qui est demandé, et s'il y a une échéance ou un impact concret."
        ),
    )
    importance: ImportanceField = Field(
        description=(
            "Niveau d'importance de l'email pour le destinataire. "
            "1 = négligeable (newsletter, promo, notification automatique sans action) ; "
            "2 = faible (information utile, aucune action attendue) ; "
            "3 = modéré (réponse ou action attendue, sans urgence) ; "
            "4 = élevé (action requise à court terme, interlocuteur clé) ; "
            "5 = critique (urgence, impact financier/juridique/sécurité, échéance imminente)."
        ),
    )
    summary: list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Résumé factuel de l'email en 1 à 3 lignes, une phrase courte par élément. "
            "Mettre en avant l'action attendue et les dates éventuelles. Rédigé en français."
        ),
    )

    @field_validator("summary")
    @classmethod
    def _clean_summary_lines(cls, lines: list[str]) -> list[str]:
        cleaned = [line.strip() for line in lines if line.strip()]
        if not cleaned:
            raise ValueError("Le résumé doit contenir au moins une ligne non vide.")
        return cleaned


class AnalyzedEmail(BaseModel):
    """Résultat exposé par l'API : métadonnées de l'email + analyse du LLM.

    Les métadonnées (id, expéditeur...) viennent de `EmailInput`, jamais du LLM :
    on ne demande pas au modèle de recopier des identifiants qu'il pourrait altérer.
    """

    email_id: str
    sender: str
    subject: str
    received_at: datetime | None
    analysis: EmailAnalysisResult

    @classmethod
    def from_email(cls, email: EmailInput, analysis: EmailAnalysisResult) -> "AnalyzedEmail":
        return cls(
            email_id=email.id,
            sender=email.sender,
            subject=email.subject,
            received_at=email.received_at,
            analysis=analysis,
        )


class AnalysisFailure(BaseModel):
    email_id: str
    error: str = Field(description="Message d'erreur lisible (timeout LLM, réponse refusée...).")


class EmailBatchAnalysisResponse(BaseModel):
    """Réponse d'une analyse par lot.

    Un échec sur un email (ex: timeout du LLM) ne doit pas faire perdre le travail
    déjà fait sur les autres : les succès et les échecs sont renvoyés séparément.
    """

    analyzed: list[AnalyzedEmail]
    failed: list[AnalysisFailure]

    def sorted_by_priority(self) -> "EmailBatchAnalysisResponse":
        """Copie triée : importance décroissante, puis du plus récent au plus ancien."""

        def priority(item: AnalyzedEmail) -> tuple[int, float]:
            timestamp = item.received_at.timestamp() if item.received_at else float("-inf")
            return item.analysis.importance, timestamp

        return self.model_copy(update={"analyzed": sorted(self.analyzed, key=priority, reverse=True)})
