"""Signal attachment encoding shared by the live adapter and the standalone sender.

signal-cli resolves ``attachments`` in ITS OWN filesystem. In the common split-container
deployment (Hermes and signal-cli in different containers, no shared volume) a Hermes cache
path does not exist there and signal-cli answers ``AttachmentInvalidException``. signal-cli
also accepts a self-contained ``data:<mime>;filename=<name>;base64,<bytes>`` URI, which works
regardless of filesystem layout.

This module owns that conversion and the Signal attachment size limit, so the gateway adapter,
the batch image path and the out-of-process sender cannot drift.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

SIGNAL_MAX_ATTACHMENT_SIZE = 100 * 1024 * 1024  # Signal's per-attachment limit
INLINE_ATTACHMENTS_ALWAYS = "always"
INLINE_ATTACHMENTS_NEVER = "never"
_TRUTHY_INLINE_MODES = frozenset({INLINE_ATTACHMENTS_ALWAYS, "true", "1", "yes", "on"})
_FALSEY_INLINE_MODES = frozenset({INLINE_ATTACHMENTS_NEVER, "false", "0", "no", "off"})


def inline_attachments_enabled(value: object) -> bool:
    """Whether to inline attachments as data URIs. ``never`` opts a shared-volume setup back into
    path uploads; anything unrecognized warns and keeps the cross-container-safe default."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if not normalized or normalized in _TRUTHY_INLINE_MODES:
        return True
    if normalized in _FALSEY_INLINE_MODES:
        return False
    logger.warning("SIGNAL_INLINE_ATTACHMENTS is not 'always' or 'never'; using 'always'")
    return True


class AttachmentTooLarge(ValueError):
    """An attachment is over Signal's limit; ``size`` carries the measured bytes for the caller."""

    def __init__(self, size: int, max_bytes: int):
        super().__init__(f"attachment too large ({size} bytes > {max_bytes})")
        self.size = size


def attachment_size_sync(
    path: str | Path,
    *,
    max_bytes: int = SIGNAL_MAX_ATTACHMENT_SIZE,
) -> int:
    """Size of a sendable attachment. Raises ``ValueError`` when it is missing, not a regular
    file, or over Signal's limit — the one owner of the outbound size contract.

    Path mode deliberately does not probe readability: a shared-volume signal-cli container may
    legitimately read a path Hermes cannot.
    """
    file_path = Path(path)
    try:
        if not file_path.is_file():
            raise ValueError("attachment is not a regular file")
        size = file_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"attachment unavailable: {type(exc).__name__}") from exc
    if size > max_bytes:
        raise AttachmentTooLarge(size, max_bytes)
    return size


def attachment_payload_size(attachment: str) -> int:
    """Wire size of one ``attachments`` entry: bytes on disk, or the serialized base64 payload.

    Used for the upload timeout. A data URI carries base64 (4/3 of the file), so measuring the
    decoded size would under-budget the actual request.
    """
    text = str(attachment or "")
    if text.startswith("data:"):
        marker = ";base64,"
        index = text.find(marker)
        return len(text) - index - len(marker) if index >= 0 else 0
    try:
        return os.path.getsize(text)
    except OSError:
        return 0


def _safe_filename(path: Path, filename: Optional[str]) -> str:
    """Percent-encode the display name: signal-cli URL-decodes data-URI parameters, so a raw
    ``%`` is invalid and ``;``/``,``/spaces would break RFC 2397 parsing or the shown name."""
    name = Path(filename).name if filename else path.name
    return quote(name, safe="") or "attachment"


def attachment_data_uri_sync(
    path: str | Path,
    *,
    filename: Optional[str] = None,
    max_bytes: int = SIGNAL_MAX_ATTACHMENT_SIZE,
) -> str:
    """Encode one local file as a signal-cli data URI. Raises ``ValueError`` when it cannot be sent.

    The caller owns user-facing error wording; this function never logs the bytes or the name
    (a filename can be PII).
    """
    file_path = Path(path)
    attachment_size_sync(file_path, max_bytes=max_bytes)
    mime = (mimetypes.guess_type(str(filename))[0] if filename else None) \
        or mimetypes.guess_type(file_path.name)[0] \
        or "application/octet-stream"
    try:
        payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ValueError(f"attachment unavailable: {type(exc).__name__}") from exc
    return f"data:{mime};filename={_safe_filename(file_path, filename)};base64,{payload}"


async def attachment_data_uri(
    path: str | Path,
    *,
    filename: Optional[str] = None,
    max_bytes: int = SIGNAL_MAX_ATTACHMENT_SIZE,
) -> str:
    """Off-loop :func:`attachment_data_uri_sync` (a 100 MB read must not stall the gateway)."""
    return await asyncio.to_thread(
        attachment_data_uri_sync, path, filename=filename, max_bytes=max_bytes)
