"""Conversation listings distinguish a committed API image from identical pasted text."""

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, RecentBrowseQuery


def _hint(digest):
    path = f"/old/profile/cache/documents/native-inputs/{digest}/api_{digest[:32]}.png"
    return f"[Image attached at: {path}]"


def _preview(store, session):
    return next(row.preview for row in store.sessions.list_recent_sessions(RecentBrowseQuery(limit=20))
                if row.session_id == session)


@pytest.mark.parametrize("rest", ["", "À Montréal."])
def test_same_stored_turn_uses_its_own_admission_and_original_text_parts(backend, scope_a, rest):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    digest = "1234abcd" * 8
    caption = "Où est l’été ? 📷"
    raw = f"{caption}\n\n{_hint(digest)}\n[screenshot]"
    typed = [{"type": "text", "text": caption}, {"type": "image_url", "image_url": {"url": f"retained:{digest}"}}]
    if rest:
        raw += "\n" + rest
        typed.append({"type": "text", "text": rest})
    for session, original, expected in (("photo", typed, f"{caption} {rest}".rstrip()),
                                        ("verbatim", raw, raw.replace("\n", " ")[:60] + "...")):
        store.sessions.create(NewSession(session, "api_server"))
        admission, _ = store.admissions.admit(epoch=epoch, session_id=session, principal_id="api",
                                              request_id="same-request", payload={"text": original, "api_turn_v1": {}})
        store.messages.append(session, NewMessage(role="user", content="Invisible", display_kind="hidden"))
        store.messages.append(session, NewMessage(role="user", content=raw, platform_message_id=admission.admission_id))
        store.messages.append(session, NewMessage(role="user", content="Later message"))
        assert _preview(store, session) == expected
        assert store.sessions.set_title(session, "Titre " + session)
        assert _preview(store, session) == expected
    store.sessions.create(NewSession("other", "api_server"))
    older = f"Ancienne photo\n\n{_hint(digest)}\n[screenshot]"
    store.messages.append("other", NewMessage(role="user", content=older, platform_message_id=admission.admission_id))
    assert _preview(store, "other") == "Ancienne photo"


@pytest.mark.parametrize("caption", ["Où ?", "é" * 59, "🙂" * 60, "À" * 61, "Première ligne\nDeuxième ligne"])
def test_older_photo_rows_without_an_admission_are_shaped_before_the_head_cut(backend, scope_a, caption):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("old", "api_server"))
    body = f"{caption}\n\n{_hint('a' * 64)}\n{_hint('b' * 64)}\n[screenshot]\n[screenshot detail=low]"
    store.messages.append("old", NewMessage(role="user", content=body))
    text = caption.replace("\n", " ")
    assert _preview(store, "old") == (text[:60] + "..." if len(text) > 60 else text)


@pytest.mark.parametrize("body", [
    "Pourquoi [Image attached at: /tmp/chat.png] apparaît-il ?",
    "Note\n\n[Image attached at: /tmp/chat.png]\n[screenshot]",
    "Note\n\n" + _hint("a" * 64).replace("api_aaaa", "api_bbbb") + "\n[screenshot]",
    "Note\n\n" + _hint("a" * 64).replace("cache/documents", "reports") + "\n[screenshot]",
    "Note\n\n" + _hint("a" * 64) + "\n[screenshot]\nEncore du texte.",
])
def test_legacy_literal_and_partial_receipts_are_preserved(backend, scope_a, body):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("literal", "cli"))
    store.messages.append("literal", NewMessage(role="user", content=body))
    text = body.replace("\n", " ")
    assert _preview(store, "literal") == (text[:60] + "..." if len(text) > 60 else text)
