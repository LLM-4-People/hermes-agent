"""A bounded display-only projection of typed API photos. No pixel decoding or file access."""

import json
from session_rules.media import legacy_api_photo_caption

PHOTO_PREVIEW_KEY = "api_photo_preview"
PREVIEW_HEAD_CHARS = 63
_DEFAULT_IMAGE_CAPTION = "What do you see in this image?"


def might_show_photo_receipt(value):
    return "[Image attached" in value and "[screenshot" in value


def api_photo_preview(payload):
    """Original text for a typed photo, or a literal copy of its receipt, in a bounded preview window."""
    if not isinstance(payload, dict) or not isinstance(payload.get("api_turn_v1"), dict):
        return None
    value = payload.get("text")
    if isinstance(value, str):
        return value.lstrip(" \t\r\n")[:PREVIEW_HEAD_CHARS] if might_show_photo_receipt(value) else None
    if not isinstance(value, list):
        return None
    head = ""
    image = literal = has_text = False
    for part in value:
        if not isinstance(part, dict):
            return None
        if part.get("type") == "image_url":
            url = part.get("image_url")
            if not isinstance(url, dict) or not isinstance(url.get("url"), str) or not url["url"]:
                return None
            image = True
        elif part.get("type") == "text" and isinstance(part.get("text"), str):
            text = part["text"]
            has_text = True
            if len(head) < PREVIEW_HEAD_CHARS:
                if head:
                    head += "\n"
                else:
                    text = text.lstrip(" \t\r\n")
                head += text[:PREVIEW_HEAD_CHARS - len(head)]
            literal = literal or might_show_photo_receipt(text)
        else:
            return None
    if not image and not literal:
        return None
    return head if has_text else _DEFAULT_IMAGE_CAPTION


def admitted_photo_preview(admission):
    """Read a pre-stamp API admission without handing damaged metadata to SQLite's JSON functions."""
    if admission is not None:
        try:
            return api_photo_preview(json.loads(admission))
        except (TypeError, ValueError, UnicodeError, RecursionError):
            pass
    return None


def archived_photo_preview(content, source, admission):
    if not isinstance(content, str):
        return content
    head = admitted_photo_preview(admission)
    if head is not None:
        return head
    return legacy_api_photo_caption(content) if source == "api_server" else content
