"""Fail-closed ephemeral pairing and invite ticket engine for Hermes Mobile.

Single source of truth for single-use, short-lived, high-entropy pairing tickets:
- Tickets are stored in SQLite WAL (``<default_hermes_root>/mobile_pairing.db``)
  via ``tools.mobile_sqlite.open_mobile_db`` with ``0600`` permissions.
- Only ``SHA-256(code)`` is persisted — raw ticket codes and ``API_SERVER_KEY``
  values are NEVER stored in the pairing database and NEVER logged.
- Claiming is strictly atomic inside ``BEGIN IMMEDIATE`` so concurrent races
  yield exactly one winner and reject all duplicates fail-closed with ``410``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re
import secrets
import time
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

from tools.mobile_sqlite import open_mobile_db

logger = logging.getLogger("gateway.mobile_pairing")

DEFAULT_TTL_SECONDS = 300
MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 900
MAX_ACTIVE_TICKETS_PER_PROFILE = 12
_CODE_TOKEN_BYTES = 24
_VALID_MODES = frozenset({"device", "invite"})
_TICKET_ID_RE = re.compile(r"^pair_[0-9a-f]{24}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pairing_tickets (
    ticket_id TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL UNIQUE,
    profile_slug TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL CHECK (mode IN ('device', 'invite')),
    created_by TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    consumed_at REAL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_pairing_profile_expires
    ON pairing_tickets(profile_slug, expires_at);
CREATE INDEX IF NOT EXISTS idx_pairing_code_hash
    ON pairing_tickets(code_hash);
"""


class PairingError(Exception):
    """Typed fail-closed pairing error with HTTP status and machine code."""

    def __init__(self, message: str, *, status_code: int = 400, code: str = "pairing_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def hash_pairing_code(code: str) -> str:
    """Return lowercase SHA-256 hex digest of a pairing code."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def normalize_public_host(raw_host: str) -> str:
    """Normalize an optional server base URL for ``hermes://pair`` links (strips path/credentials)."""
    cleaned = (raw_host or "").strip()
    if not cleaned:
        return ""
    if _CONTROL_CHAR_RE.search(cleaned):
        raise PairingError("Server host contains invalid characters.", status_code=400, code="invalid_host")
    candidate = cleaned if "://" in cleaned else f"https://{cleaned}"
    parsed = urlparse(candidate)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise PairingError("Server host must be a valid http(s) URL.", status_code=400, code="invalid_host")
    if parsed.username or parsed.password:
        raise PairingError("Server host must not contain embedded credentials.", status_code=400, code="invalid_host")
    if parsed.query or parsed.fragment:
        raise PairingError("Server host must not include query or fragment.", status_code=400, code="invalid_host")
    base_path = (parsed.path or "").rstrip("/")
    return f"{scheme}://{parsed.netloc}{base_path}"


def build_pairing_deep_link(host: str, code: str, *, profile_slug: str = "") -> str:
    """Build a canonical ``hermes://pair?...`` link for QR codes and 1-tap mobile onboarding."""
    params: list[tuple[str, str]] = []
    norm_host = normalize_public_host(host) if host else ""
    if norm_host:
        params.append(("host", norm_host))
    params.append(("code", code))
    slug = (profile_slug or "").strip().lower()
    if slug and slug != "default":
        params.append(("profile", slug))
    return f"hermes://pair?{urlencode(params)}"


def validate_ttl_seconds(raw_ttl: Any) -> int:
    """Clamp and validate ticket TTL strictly within ``[30, 900]`` seconds."""
    if raw_ttl is None:
        return DEFAULT_TTL_SECONDS
    if isinstance(raw_ttl, bool):
        raise PairingError("ttl_seconds must be an integer.", status_code=400, code="invalid_ttl")
    try:
        ttl = int(raw_ttl)
    except (TypeError, ValueError) as exc:
        raise PairingError("ttl_seconds must be an integer.", status_code=400, code="invalid_ttl") from exc
    if ttl < MIN_TTL_SECONDS or ttl > MAX_TTL_SECONDS:
        raise PairingError(
            f"ttl_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}.",
            status_code=400,
            code="invalid_ttl",
        )
    return ttl


def validate_mode(raw_mode: Any) -> str:
    mode = str(raw_mode or "device").strip().lower()
    if mode not in _VALID_MODES:
        raise PairingError("mode must be 'device' or 'invite'.", status_code=400, code="invalid_mode")
    return mode


def _default_root_home() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root()


class PairingStore:
    """Thread- and process-safe ephemeral pairing ticket store backed by SQLite WAL."""

    def __init__(self, root_home: Optional[Path] = None, *, clock=time.time) -> None:
        self._root_home = Path(root_home) if root_home is not None else _default_root_home()
        self._clock = clock

    @property
    def db_path(self) -> Path:
        return self._root_home / "mobile_pairing.db"

    def _open(self, *, write: bool = False):
        if write:
            self._root_home.mkdir(parents=True, exist_ok=True)
        return open_mobile_db(
            self.db_path,
            schema_key="mobile_pairing_v1",
            schema_sql=_SCHEMA_SQL,
            db_label="mobile_pairing.db",
            write=write,
        )

    def create_ticket(
        self,
        *,
        profile_slug: str,
        display_name: str = "",
        mode: str = "device",
        created_by: str = "default",
        ttl_seconds: Any = DEFAULT_TTL_SECONDS,
        host: str = "",
    ) -> dict[str, Any]:
        """Mint one single-use pairing ticket; plaintext ``code`` is returned ONLY here."""
        slug = (profile_slug or "default").strip().lower()
        clean_mode = validate_mode(mode)
        ttl = validate_ttl_seconds(ttl_seconds)
        norm_host = normalize_public_host(host) if host else ""
        clean_display = (display_name or slug).strip()[:80]
        if _CONTROL_CHAR_RE.search(clean_display):
            raise PairingError("display_name contains control characters.", status_code=400, code="invalid_display_name")

        now = float(self._clock())
        expires_at = now + ttl
        ticket_id = f"pair_{secrets.token_hex(12)}"
        code = secrets.token_urlsafe(_CODE_TOKEN_BYTES)
        code_digest = hash_pairing_code(code)

        with self._open(write=True) as conn:
            # Prune expired rows older than 24h to keep the database bounded.
            conn.execute("DELETE FROM pairing_tickets WHERE expires_at < ?", (now - 86_400.0,))
            active_row = conn.execute(
                """
                SELECT COUNT(*) AS cnt FROM pairing_tickets
                WHERE profile_slug = ? AND consumed_at IS NULL AND revoked_at IS NULL AND expires_at > ?
                """,
                (slug, now),
            ).fetchone()
            if active_row and int(active_row["cnt"]) >= MAX_ACTIVE_TICKETS_PER_PROFILE:
                raise PairingError(
                    "Too many active pairing tickets for this profile. Revoke an unused ticket first.",
                    status_code=429,
                    code="too_many_active_tickets",
                )
            conn.execute(
                """
                INSERT INTO pairing_tickets (
                    ticket_id, code_hash, profile_slug, display_name, mode,
                    created_by, created_at, expires_at, consumed_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (ticket_id, code_digest, slug, clean_display, clean_mode, created_by, now, expires_at),
            )

        logger.info("Minted pairing ticket %s for profile=%r mode=%s ttl=%ds", ticket_id, slug, clean_mode, ttl)
        return {
            "ticket_id": ticket_id,
            "code": code,
            "expires_at": expires_at,
            "ttl_seconds": ttl,
            "deep_link": build_pairing_deep_link(norm_host, code, profile_slug=slug),
            "profile_slug": slug,
            "display_name": clean_display,
            "mode": clean_mode,
        }

    def claim_ticket(self, code: Any) -> dict[str, Any]:
        """Atomically consume one unused, non-expired pairing code (single-use guaranteed)."""
        if not isinstance(code, str) or not code.strip():
            raise PairingError("Pairing code is required.", status_code=400, code="missing_pairing_code")
        cleaned = code.strip()
        if len(cleaned) < 12 or len(cleaned) > 256 or _CONTROL_CHAR_RE.search(cleaned):
            raise PairingError("Invalid pairing code format.", status_code=400, code="invalid_pairing_code")

        digest = hash_pairing_code(cleaned)
        now = float(self._clock())

        with self._open(write=True) as conn:
            row = conn.execute(
                """
                SELECT ticket_id, profile_slug, display_name, mode, created_by,
                       created_at, expires_at, consumed_at, revoked_at
                FROM pairing_tickets
                WHERE code_hash = ?
                """,
                (digest,),
            ).fetchone()
            if row is None:
                raise PairingError("Pairing ticket not found.", status_code=404, code="pairing_ticket_not_found")
            if row["revoked_at"] is not None:
                raise PairingError("Pairing ticket was revoked.", status_code=410, code="pairing_ticket_revoked")
            if row["consumed_at"] is not None:
                raise PairingError("Pairing ticket was already used.", status_code=410, code="pairing_ticket_consumed")
            if float(row["expires_at"]) <= now:
                raise PairingError("Pairing ticket has expired.", status_code=410, code="pairing_ticket_expired")

            cur = conn.execute(
                """
                UPDATE pairing_tickets
                SET consumed_at = ?
                WHERE ticket_id = ? AND consumed_at IS NULL AND revoked_at IS NULL AND expires_at > ?
                """,
                (now, row["ticket_id"], now),
            )
            if cur.rowcount != 1:
                raise PairingError("Pairing ticket was already used.", status_code=410, code="pairing_ticket_consumed")

        logger.info("Consumed pairing ticket %s for profile=%r", row["ticket_id"], row["profile_slug"])
        return {
            "ticket_id": row["ticket_id"],
            "profile_slug": row["profile_slug"],
            "display_name": row["display_name"],
            "mode": row["mode"],
            "created_by": row["created_by"],
            "created_at": float(row["created_at"]),
            "expires_at": float(row["expires_at"]),
            "consumed_at": now,
        }

    def list_active_tickets(self, *, profile_slug: Optional[str] = None) -> list[dict[str, Any]]:
        """List non-expired, unconsumed, unrevoked tickets (metadata only — never returns secrets)."""
        now = float(self._clock())
        with self._open(write=False) as conn:
            if profile_slug is not None:
                slug = profile_slug.strip().lower()
                rows = conn.execute(
                    """
                    SELECT ticket_id, profile_slug, display_name, mode, created_by, created_at, expires_at
                    FROM pairing_tickets
                    WHERE profile_slug = ? AND consumed_at IS NULL AND revoked_at IS NULL AND expires_at > ?
                    ORDER BY created_at DESC
                    """,
                    (slug, now),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT ticket_id, profile_slug, display_name, mode, created_by, created_at, expires_at
                    FROM pairing_tickets
                    WHERE consumed_at IS NULL AND revoked_at IS NULL AND expires_at > ?
                    ORDER BY created_at DESC
                    """,
                    (now,),
                ).fetchall()
        return [
            {
                "ticket_id": r["ticket_id"],
                "profile_slug": r["profile_slug"],
                "display_name": r["display_name"],
                "mode": r["mode"],
                "created_by": r["created_by"],
                "created_at": float(r["created_at"]),
                "expires_at": float(r["expires_at"]),
                "ttl_remaining_seconds": max(0, int(float(r["expires_at"]) - now)),
            }
            for r in rows
        ]

    def revoke_ticket(
        self,
        ticket_id: str,
        *,
        allowed_profile: Optional[str] = None,
        is_admin: bool = False,
    ) -> dict[str, Any]:
        """Revoke an active unused ticket immediately; enforces profile ownership unless admin."""
        clean_id = (ticket_id or "").strip()
        if not _TICKET_ID_RE.fullmatch(clean_id):
            raise PairingError("Invalid ticket_id.", status_code=400, code="invalid_ticket_id")
        now = float(self._clock())
        with self._open(write=True) as conn:
            row = conn.execute(
                """
                SELECT ticket_id, profile_slug, consumed_at, revoked_at, expires_at
                FROM pairing_tickets
                WHERE ticket_id = ?
                """,
                (clean_id,),
            ).fetchone()
            if row is None:
                raise PairingError("Pairing ticket not found.", status_code=404, code="pairing_ticket_not_found")
            if not is_admin and allowed_profile is not None:
                if row["profile_slug"] != allowed_profile.strip().lower():
                    raise PairingError(
                        "You can only revoke pairing tickets for your own profile.",
                        status_code=403,
                        code="profile_access_denied",
                    )
            if row["consumed_at"] is not None:
                raise PairingError("Pairing ticket was already used.", status_code=410, code="pairing_ticket_consumed")
            if row["revoked_at"] is not None:
                return {"ticket_id": clean_id, "profile_slug": row["profile_slug"], "revoked": True}
            conn.execute(
                "UPDATE pairing_tickets SET revoked_at = ? WHERE ticket_id = ?",
                (now, clean_id),
            )
        logger.info("Revoked pairing ticket %s for profile=%r", clean_id, row["profile_slug"])
        return {"ticket_id": clean_id, "profile_slug": row["profile_slug"], "revoked": True}

    def revoke_all_for_profile(self, profile_slug: str) -> int:
        """Revoke every outstanding unused ticket for *profile_slug* (used on key rotation/delete)."""
        slug = (profile_slug or "").strip().lower()
        if not slug:
            return 0
        now = float(self._clock())
        with self._open(write=True) as conn:
            cur = conn.execute(
                """
                UPDATE pairing_tickets
                SET revoked_at = ?
                WHERE profile_slug = ? AND consumed_at IS NULL AND revoked_at IS NULL
                """,
                (now, slug),
            )
            return int(cur.rowcount or 0)
