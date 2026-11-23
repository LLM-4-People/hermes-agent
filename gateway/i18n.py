"""Centralized bilingual (EN / FR) localization and anti-AI-slop SSoT for Hermes gateway & mobile tools.

Provides:
- Deterministic per-request language resolution (``X-Hermes-Lang`` -> explicit ``lang`` ->
  ``Accept-Language`` -> profile/session language -> lightweight FR/EN text heuristic -> ``en``).
- Concurrency-safe ``ContextVar`` scoping (``user_lang_scope``) so simultaneous users
  (e.g., Alice in French and Bob in English) receive pure single-language replies with zero cross-leak.
- Flat dotted catalog ``CATALOG`` with 100% EN/FR key and placeholder parity, short natural human prose,
  and zero AI slop.
- Reusable anti-AI-slop detector helpers used by both prompt guidance and automated CI tests.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import contextvars
import math
import re
from typing import Any, Optional

SUPPORTED_USER_LANGS: frozenset[str] = frozenset({"en", "fr"})
DEFAULT_USER_LANG: str = "en"

_current_user_lang: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "hermes_gateway_user_lang",
    default=None,
)

_LANG_ALIASES: dict[str, str] = {
    "en": "en",
    "english": "en",
    "anglais": "en",
    "fr": "fr",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
}

_FRENCH_DIACRITICS_RE = re.compile(r"[àâçéèêëîïôùûüœæÀÂÇÉÈÊËÎÏÔÙÛÜŒÆ]")
_WORD_RE = re.compile(r"[a-zA-ZàâçéèêëîïôùûüœæÀÂÇÉÈÊËÎÏÔÙÛÜŒÆ']+")

_FRENCH_MARKERS: frozenset[str] = frozenset({
    "bonjour", "merci", "salut", "avec", "pour", "dans", "mais", "pas", "une", "des",
    "les", "est", "sont", "vous", "nous", "cette", "mon", "mes", "votre", "peux",
    "comment", "pourquoi", "quoi", "quel", "quelle", "aujourd'hui", "demain",
    "affiche", "montre", "crée", "créer", "résume", "explique", "corrige", "aide",
})
_ENGLISH_MARKERS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "have", "what", "when",
    "where", "which", "please", "thanks", "hello", "show", "create", "summarize",
    "explain", "fix", "help", "today", "tomorrow", "your", "should", "would",
})


def normalize_user_lang(raw: Any) -> Optional[str]:
    """Normalize a language tag or alias to ``'fr'`` or ``'en'``, else ``None``."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.split(";", 1)[0].strip().lower().replace("_", "-")
    if not cleaned:
        return None
    if cleaned in _LANG_ALIASES:
        return _LANG_ALIASES[cleaned]
    primary = cleaned.split("-", 1)[0]
    return _LANG_ALIASES.get(primary)


def parse_accept_language_header(header: Optional[str]) -> Optional[str]:
    """Extract highest-priority ``'fr'`` or ``'en'`` from an HTTP ``Accept-Language`` header."""
    if not isinstance(header, str) or not header.strip():
        return None
    ranked: list[tuple[float, int, str]] = []
    for idx, entry in enumerate(header.split(",")):
        part = entry.strip()
        if not part:
            continue
        lang = normalize_user_lang(part)
        if lang is None:
            continue
        q = 1.0
        for param in part.split(";")[1:]:
            k, _, v = param.strip().partition("=")
            if k.strip().lower() == "q":
                try:
                    q = float(v.strip())
                except ValueError:
                    q = 0.0
        if math.isfinite(q) and 0.0 < q <= 1.0:
            ranked.append((-q, idx, lang))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][2]


def detect_text_lang(text: Optional[str]) -> Optional[str]:
    """Lightweight French/English heuristic on natural user text; returns ``None`` when ambiguous."""
    if not isinstance(text, str):
        return None
    sample = text.strip()[:600]
    if len(sample) < 4:
        return None
    words = [w.lower() for w in _WORD_RE.findall(sample)]
    if not words:
        return None
    fr_score = sum(2 for w in words if w in _FRENCH_MARKERS)
    en_score = sum(2 for w in words if w in _ENGLISH_MARKERS)
    if _FRENCH_DIACRITICS_RE.search(sample):
        fr_score += 3
    if fr_score >= 3 and fr_score > en_score:
        return "fr"
    if en_score >= 3 and en_score > fr_score:
        return "en"
    return None


def resolve_user_lang(
    accept_language: Optional[str] = None,
    explicit_lang: Optional[str] = None,
    text: Optional[str] = None,
    *,
    profile_lang: Optional[str] = None,
    default: str = DEFAULT_USER_LANG,
) -> str:
    """Resolve user language deterministically: explicit -> Accept-Language -> profile -> text -> default."""
    for candidate in (
        normalize_user_lang(explicit_lang),
        parse_accept_language_header(accept_language),
        normalize_user_lang(profile_lang),
        detect_text_lang(text),
    ):
        if candidate in SUPPORTED_USER_LANGS:
            return candidate
    fallback = normalize_user_lang(default)
    return fallback if fallback in SUPPORTED_USER_LANGS else DEFAULT_USER_LANG


def resolve_request_lang(
    headers: Optional[Mapping[str, str]],
    *,
    explicit_lang: Optional[str] = None,
    profile_lang: Optional[str] = None,
    text: Optional[str] = None,
    default: str = DEFAULT_USER_LANG,
) -> str:
    """Extract ``X-Hermes-Lang`` / ``Accept-Language`` from HTTP headers and resolve ``'fr'`` or ``'en'``."""
    header_explicit: Optional[str] = None
    accept_language: Optional[str] = None
    if headers is not None:
        # Case-insensitive lookup works with aiohttp CIMultiDictProxy and plain dicts alike.
        for k, v in headers.items():
            kl = str(k).lower()
            if kl == "x-hermes-lang" and v:
                header_explicit = str(v)
            elif kl == "accept-language" and v:
                accept_language = str(v)
    return resolve_user_lang(
        accept_language=accept_language,
        explicit_lang=normalize_user_lang(explicit_lang) or header_explicit,
        profile_lang=profile_lang,
        text=text,
        default=default,
    )


def active_scoped_lang() -> Optional[str]:
    """Return the active per-request scoped language if set, else ``None``."""
    return _current_user_lang.get()


def current_user_lang(default: str = DEFAULT_USER_LANG) -> str:
    """Return the active user language from the ContextVar, falling back to ``default``."""
    scoped = _current_user_lang.get()
    if scoped in SUPPORTED_USER_LANGS:
        return scoped
    fallback = normalize_user_lang(default)
    return fallback if fallback in SUPPORTED_USER_LANGS else DEFAULT_USER_LANG


@contextmanager
def user_lang_scope(lang: Optional[str]) -> Iterator[str]:
    """Bind the active user language for the current async task or thread cleanly."""
    resolved = normalize_user_lang(lang) or DEFAULT_USER_LANG
    token = _current_user_lang.set(resolved)
    try:
        yield resolved
    finally:
        _current_user_lang.reset(token)


# Single Source of Truth bilingual catalog: 100% EN/FR key and placeholder parity,
# short direct human prose, zero AI slop, zero mixed languages.
CATALOG: dict[str, dict[str, str]] = {
    "en": {
        # Gateway slash commands
        "cmd.help.header": "Available commands:",
        "cmd.help.new": "/new — start a new conversation",
        "cmd.help.reset": "/reset — clear the current conversation",
        "cmd.help.status": "/status — show active model and session state",
        "cmd.help.stop": "/stop — stop the current reply",
        "cmd.help.model": "/model — view or switch the active model",
        "cmd.help.usage": "/usage — show token and cost totals",
        "cmd.help.compress": "/compress — compact earlier messages",
        "cmd.help.title": "/title — rename this conversation",
        "cmd.help.retry": "/retry — run the last message again",
        "cmd.help.undo": "/undo — remove the last exchange",
        "cmd.help.personality": "/personality — view or switch writing style",
        "cmd.new.done": "Started a new conversation.",
        "cmd.reset.done": "Conversation cleared.",
        "cmd.stop.done": "Stopped the active task.",
        "cmd.stop.idle": "No active task to stop.",
        "cmd.status.summary": "Model: {model}. Messages in thread: {count}.",
        "cmd.model.current": "Current model: {model}.",
        "cmd.model.switched": "Model switched to {model}.",
        "cmd.usage.summary": "Tokens used: {tokens}.",
        "cmd.compress.done": "Earlier messages compacted.",
        "cmd.title.updated": "Title updated to: {title}.",
        "cmd.retry.empty": "No previous message to retry.",
        "cmd.undo.done": "Removed the last exchange.",
        "cmd.undo.empty": "Nothing to undo in this conversation.",
        "cmd.personality.current": "Current style: {name}.",
        # Gateway progress, status & error notices
        "gateway.progress.working": "Working on your request...",
        "gateway.progress.retrying": "Connection interrupted. Retrying...",
        "gateway.error.timeout": "The request timed out. Try again or shorten the ask.",
        "gateway.error.unauthorized": "Access denied. Check the profile API key.",
        "gateway.error.rate_limited": "Too many requests right now. Wait a moment and try again.",
        "gateway.error.unavailable": "The assistant could not complete this request right now.",
        "gateway.error.busy": "A task is already running in this conversation.",
        # API routes (OpenAI & shared HTTP errors)
        "api.invalid_json": "Invalid JSON in request body",
        "api.missing_messages": "Missing or invalid 'messages' field",
        "api.no_user_message": "No user message found in messages",
        "api.session_auth_required": "Session continuation requires API key authentication. Configure API_SERVER_KEY to enable this feature.",
        "api.invalid_session_id": "Invalid session ID",
        "api.session_id_too_long": "Session ID too long",
        "api.missing_input": "Missing 'input' field",
        "api.invalid_input": "'input' must be a string or array",
        "api.invalid_history": "'conversation_history' must be an array of message objects",
        "api.invalid_history_item": "conversation_history[{index}] must have 'role' and 'content' fields",
        "api.conflicting_conversation": "Cannot use both 'conversation' and 'previous_response_id'",
        "api.response_not_found": "Response not found: {response_id}",
        "api.internal_error": "Internal server error: {detail}",
        # Habits API
        "habits.invalid_key_empty": "'{field}' must be a non-empty lowercase token.",
        "habits.invalid_key_format": "'{field}' must match lowercase slug format (a-z, 0-9, '.', '_', '-', max 64 chars).",
        "habits.invalid_int_query": "limit and min_count must be integers.",
        "habits.invalid_limit": "limit must be 1..{max_limit}.",
        "habits.disallowed_fields": "Only anonymized 'category' and 'action' fields are allowed (rejected: {fields}).",
        # Clarify & Approval
        "clarify.unavailable": "Clarify tool is not available in this execution context.",
        "clarify.failed": "Failed to get user input: {detail}",
        "clarify.recommended": "(Recommended)",
        "approval.required_notice": "⚠️ This action is potentially dangerous ({description}). Asking the user for approval.\n\n**Target:**\n```\n{command}\n```",
        "approval.pending_notice": "⚠️ {description}. Asking the user for approval.\n\n{body}\n\nSTOP: do NOT re-run, rephrase, or re-issue this {noun} — each variant sends the user ANOTHER approval card. Wait for the user's decision; if this turn must end, report that approval is pending.",
        # Mobile tools (Notes, Sharing, Deliver)
        "mobile.invalid_parameter": "Invalid parameter value.",
        "mobile.storage_error": "Storage operation failed.",
        "mobile.deliver_hint": "Use kind in {kinds} (alias 'alert'->'notice'), urgency in {urgencies} (alias 'low'->'silent'), non-empty title, and JSON payload.",
        "mobile.deliver_example_title": "Morning briefing",
        "mobile.deliver_example_summary": "3 priority tasks for today",
    },
    "fr": {
        # Gateway slash commands
        "cmd.help.header": "Commandes disponibles :",
        "cmd.help.new": "/new — ouvrir une nouvelle discussion",
        "cmd.help.reset": "/reset — vider la discussion en cours",
        "cmd.help.status": "/status — afficher le modèle actif et l’état du fil",
        "cmd.help.stop": "/stop — arrêter la réponse en cours",
        "cmd.help.model": "/model — voir ou changer le modèle actif",
        "cmd.help.usage": "/usage — afficher la consommation et le coût",
        "cmd.help.compress": "/compress — résumer les messages précédents",
        "cmd.help.title": "/title — renommer cette discussion",
        "cmd.help.retry": "/retry — relancer le dernier message",
        "cmd.help.undo": "/undo — retirer le dernier échange",
        "cmd.help.personality": "/personality — voir ou changer le ton",
        "cmd.new.done": "Nouvelle discussion ouverte.",
        "cmd.reset.done": "Discussion vidée.",
        "cmd.stop.done": "Tâche en cours arrêtée.",
        "cmd.stop.idle": "Aucune tâche active à arrêter.",
        "cmd.status.summary": "Modèle : {model}. Messages dans ce fil : {count}.",
        "cmd.model.current": "Modèle actif : {model}.",
        "cmd.model.switched": "Modèle changé pour {model}.",
        "cmd.usage.summary": "Volume consommé : {tokens}.",
        "cmd.compress.done": "Messages précédents compactés.",
        "cmd.title.updated": "Titre mis à jour : {title}.",
        "cmd.retry.empty": "Aucun message précédent à relancer.",
        "cmd.undo.done": "Dernier échange retiré.",
        "cmd.undo.empty": "Rien à annuler dans cette discussion.",
        "cmd.personality.current": "Ton actif : {name}.",
        # Gateway progress, status & error notices
        "gateway.progress.working": "Traitement de votre demande en cours...",
        "gateway.progress.retrying": "Connexion interrompue. Nouvelle tentative...",
        "gateway.error.timeout": "Le délai d’attente est dépassé. Réessayez ou raccourcissez la demande.",
        "gateway.error.unauthorized": "Accès refusé. Vérifiez la clé API de ce profil.",
        "gateway.error.rate_limited": "Trop de demandes en ce moment. Patientez quelques secondes avant de réessayer.",
        "gateway.error.unavailable": "L’assistant n’a pas pu terminer cette demande pour le moment.",
        "gateway.error.busy": "Une tâche est déjà en cours dans cette discussion.",
        # API routes (OpenAI & shared HTTP errors)
        "api.invalid_json": "Corps de requête JSON invalide.",
        "api.missing_messages": "Champ « messages » absent ou invalide.",
        "api.no_user_message": "Aucun message utilisateur trouvé dans « messages ».",
        "api.session_auth_required": "La reprise de session exige une clé API. Configurez API_SERVER_KEY pour activer cette option.",
        "api.invalid_session_id": "Identifiant de session invalide.",
        "api.session_id_too_long": "Identifiant de session trop long.",
        "api.missing_input": "Champ « input » manquant.",
        "api.invalid_input": "Le champ « input » doit être une chaîne ou une liste.",
        "api.invalid_history": "Le champ « conversation_history » doit être une liste de messages.",
        "api.invalid_history_item": "conversation_history[{index}] doit contenir les champs « role » et « content ».",
        "api.conflicting_conversation": "Impossible d’utiliser « conversation » et « previous_response_id » en même temps.",
        "api.response_not_found": "Réponse introuvable : {response_id}.",
        "api.internal_error": "Erreur interne du serveur : {detail}",
        # Habits API
        "habits.invalid_key_empty": "Le champ « {field} » doit être un identifiant minuscule non vide.",
        "habits.invalid_key_format": "Le champ « {field} » doit utiliser uniquement a-z, 0-9, '.', '_', '-' (64 caractères max).",
        "habits.invalid_int_query": "Les paramètres limit et min_count doivent être des entiers.",
        "habits.invalid_limit": "Le paramètre limit doit être compris entre 1 et {max_limit}.",
        "habits.disallowed_fields": "Seuls les champs anonymisés « category » et « action » sont acceptés (rejetés : {fields}).",
        # Clarify & Approval
        "clarify.unavailable": "L’outil de clarification n’est pas disponible dans ce contexte.",
        "clarify.failed": "Impossible de récupérer la réponse de l’utilisateur : {detail}",
        "clarify.recommended": "(Recommandé)",
        "approval.required_notice": "⚠️ Cette action est sensible ({description}). Demande d’autorisation envoyée à l’utilisateur.\n\n**Cible :**\n```\n{command}\n```",
        "approval.pending_notice": "⚠️ {description}. Demande d’autorisation envoyée à l’utilisateur.\n\n{body}\n\nARRÊT : ne relancez pas et ne reformulez pas cette action ({noun}) ; attendez la décision de l’utilisateur.",
        # Mobile tools (Notes, Sharing, Deliver)
        "mobile.invalid_parameter": "Valeur de paramètre invalide.",
        "mobile.storage_error": "Échec de l’opération de stockage.",
        "mobile.deliver_hint": "Utilisez kind dans {kinds} (alias 'alert'->'notice'), urgency dans {urgencies} (alias 'low'->'silent'), un titre non vide et un objet JSON.",
        "mobile.deliver_example_title": "Point du matin",
        "mobile.deliver_example_summary": "3 tâches prioritaires pour aujourd’hui",
    },
}


def t(key: str, lang: Optional[str] = None, **kwargs: Any) -> str:
    """Translate ``key`` into ``lang`` (or active ``current_user_lang()``) with safe formatting."""
    target = normalize_user_lang(lang) if lang is not None else current_user_lang()
    if target not in CATALOG:
        target = DEFAULT_USER_LANG
    template = CATALOG[target].get(key)
    if template is None:
        template = CATALOG[DEFAULT_USER_LANG].get(key, key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


# --- Anti-AI-Slop style guidance & linter rules (Wikipedia "Signs of AI Writing") ---

BANNED_SLOP_PHRASES_EN: tuple[str, ...] = (
    "great question",
    "certainly!",
    "i'd be happy to",
    "i would be happy to",
    "of course!",
    "let me know if you need anything else",
    "feel free to ask",
    "hope this helps",
    "it's not just",
    "not just x, it's",
    "game-changer",
    "in today's fast-paced",
)

BANNED_SLOP_PHRASES_FR: tuple[str, ...] = (
    "excellente question",
    "bien sûr !",
    "avec plaisir !",
    "n'hésitez pas si vous avez d'autres questions",
    "n’hésitez pas si vous avez d’autres questions",
    "j'espère que cela vous aide",
    "paysage numérique",
    "ce n'est pas seulement",
)

BANNED_SLOP_WORDS_EN: tuple[str, ...] = (
    "delve",
    "tapestry",
    "testament",
    "landscape",
    "pivotal",
    "beacon",
    "foster",
    "elevate",
    "unlock",
    "seamlessly",
    "robust",
    "holistic",
    "leverage",
    "empower",
)

BANNED_SLOP_WORDS_FR: tuple[str, ...] = (
    "tapisserie",
    "incontournable",
    "synergie",
    "holistique",
    "levier",
    "propulser",
    "écosystème numérique",
)


def find_slop_hits(text: str, lang: str = "en") -> list[str]:
    """Return any banned AI-slop phrases or cliché words found in ``text``."""
    if not text:
        return []
    lower = text.lower()
    phrases = BANNED_SLOP_PHRASES_FR if lang == "fr" else BANNED_SLOP_PHRASES_EN
    words = BANNED_SLOP_WORDS_FR if lang == "fr" else BANNED_SLOP_WORDS_EN
    hits: list[str] = []
    for phrase in phrases:
        if phrase in lower:
            hits.append(phrase)
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            hits.append(word)
    return hits


ANTI_SLOP_PROMPT_RULES = (
    "Write in natural, plain prose matching the user's language (French or English, never mixed). "
    "Cut sycophantic openers and robotic sign-offs, cliché buzzwords (e.g. delve, tapestry, testament, "
    "pivotal, holistic, seamlessly, leverage, tapisserie, synergie, incontournable), negative parallelisms "
    "('not just X, it's Y'), forced rule-of-three lists, false ranges, and em-dash overuse."
)


__all__ = [
    "ANTI_SLOP_PROMPT_RULES",
    "BANNED_SLOP_PHRASES_EN",
    "BANNED_SLOP_PHRASES_FR",
    "BANNED_SLOP_WORDS_EN",
    "BANNED_SLOP_WORDS_FR",
    "CATALOG",
    "DEFAULT_USER_LANG",
    "SUPPORTED_USER_LANGS",
    "active_scoped_lang",
    "current_user_lang",
    "detect_text_lang",
    "find_slop_hits",
    "normalize_user_lang",
    "parse_accept_language_header",
    "resolve_request_lang",
    "resolve_user_lang",
    "t",
    "user_lang_scope",
]
