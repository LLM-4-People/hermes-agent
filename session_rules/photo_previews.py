"""A bounded display-only projection of typed API photos. No pixel decoding or file access."""

import json
from session_rules.media import legacy_api_photo_caption

PHOTO_PREVIEW_KEY = "api_photo_preview"
PREVIEW_HEAD_CHARS = 63
_DEFAULT_IMAGE_CAPTION = "What do you see in this image?"


def valid_photo_head(value):
    if not isinstance(value, str):
        return None
    head = value[:PREVIEW_HEAD_CHARS]
    try:
        head.encode("utf-8")
    except UnicodeEncodeError:
        # Historical JSON can legally escape an isolated surrogate; SQLite TEXT cannot carry it.
        return None
    return head


def _json_object(raw):
    if raw is not None:
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                return value
        except (TypeError, ValueError, RecursionError):
            pass
    return {}


def stored_photo_preview(metadata):
    return valid_photo_head(_json_object(metadata).get(PHOTO_PREVIEW_KEY))


def might_show_photo_receipt(value):
    return "[Image attached" in value and "[screenshot" in value


def api_photo_preview(payload):
    """Original text for a typed photo, or a literal copy of its receipt, in a bounded preview window."""
    if not isinstance(payload, dict) or not isinstance(payload.get("api_turn_v1"), dict):
        return None
    value = payload.get("text")
    if isinstance(value, str):
        return valid_photo_head(value.lstrip(" \t\r\n")) if might_show_photo_receipt(value) else None
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
    return valid_photo_head(head) if has_text else _DEFAULT_IMAGE_CAPTION


def admitted_photo_preview(admission):
    """Read a pre-stamp API admission without handing damaged metadata to SQLite's JSON functions."""
    return api_photo_preview(_json_object(admission))


def archived_photo_preview(content, source, admission):
    if not isinstance(content, str):
        return content
    head = admitted_photo_preview(admission)
    if head is not None:
        return head
    return legacy_api_photo_caption(content) if source == "api_server" else content
