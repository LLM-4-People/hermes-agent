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


@pytest.mark.parametrize("source", ["cli", "telegram"])
def test_only_api_history_can_assume_an_unlinked_canonical_suffix(backend, scope_a, source):
    store = backend.open(scope_a)
    body = f"Vérifier.\n\n{_hint('c' * 64)}\n[screenshot]"
    store.sessions.create(NewSession("ordinary", source))
    store.messages.append("ordinary", NewMessage(role="user", content=body))
    assert _preview(store, "ordinary") == body.replace("\n", " ")[:60] + "..."


@pytest.mark.parametrize("bad", ["{", '["broken"]', '{"api_turn_v1":{},"text":[null]}',
                                 '{"api_turn_v1":{},"text":[{"type":"text","text":{"private":"not a caption"}}]}'])
def test_damaged_admission_cannot_block_other_sessions_or_writes(backend, scope_a, bad):
    import sqlite3
    from tests.persistence_sqlite.conftest import store_for

    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    for sid in ("damaged", "healthy"):
        store.sessions.create(NewSession(sid, "api_server"))
    admission, _ = store.admissions.admit(epoch=epoch, session_id="damaged", principal_id="api",
                                          request_id="broken", payload={"text": "before"})
    with sqlite3.connect(store_for(scope_a)) as conn:
        conn.execute("UPDATE session_admissions SET payload_json=? WHERE admission_id=?", (bad, admission.admission_id))
    store.messages.append("damaged", NewMessage(role="user", content=f"Photo\n\n{_hint('d' * 64)}\n[screenshot]",
                                                platform_message_id=admission.admission_id))
    store.messages.append("healthy", NewMessage(role="user", content="Bonjour."))
    assert _preview(store, "damaged") == "Photo"
    assert _preview(store, "healthy") == "Bonjour."
    assert store.sessions.set_title("healthy", "Après la lecture")


@pytest.mark.parametrize("parts, expected", [
    ([{"type": "text", "text": "Où ?"}], "Où ?"), ([], "What do you see in this image?"),
])
def test_linked_remote_image_uses_its_original_caption(backend, scope_a, parts, expected):
    store = backend.open(scope_a)
    store.sessions.create(NewSession("remote", "api_server"))
    epoch = store.runtime.begin_epoch(instance_id="owner")
    url = "https://example.invalid/sample.png"
    typed = [*parts, {"type": "image_url", "image_url": {"url": url}}]
    admission, _ = store.admissions.admit(epoch=epoch, session_id="remote", principal_id="api",
                                          request_id="photo", payload={"text": typed, "api_turn_v1": {}})
    store.messages.append("remote", NewMessage(role="user", content=f"{expected}\n\n[Image attached: {url}]\n[screenshot]",
                                               platform_message_id=admission.admission_id))
    assert _preview(store, "remote") == expected


def test_export_of_linked_historical_literal_preserves_its_preview_without_transferring_admissions(backend, scope_a, scope_b):
    from hermes_cli.session_export import to_legacy_export_dict

    source, target = backend.open(scope_a), backend.open(scope_b)
    source.sessions.create(NewSession("portable", "api_server"))
    epoch = source.runtime.begin_epoch(instance_id="owner")
    body = f"Exemple\n\n{_hint('e' * 64)}\n[screenshot]"
    owner, _ = source.admissions.admit(epoch=epoch, session_id="portable", principal_id="api", request_id="literal",
                                       payload={"text": body, "api_turn_v1": {}})
    source.messages.append("portable", NewMessage(role="user", content=body, platform_message_id=owner.admission_id))
    target.maintenance.import_sessions([to_legacy_export_dict(source.maintenance.export_session("portable"))])
    assert _preview(target, "portable") == _preview(source, "portable") == body.replace("\n", " ")[:60] + "..."
    assert target.maintenance.export_session("portable").messages[0].content == body



def test_stamped_caption_survives_ledger_removal_and_is_bounded_to_its_own_event(backend, scope_a):
    from types import SimpleNamespace
    from gateway.session_display import admission_submission, submission_turn_scope
    from session_rules.photo_previews import PHOTO_PREVIEW_KEY, api_photo_preview

    source = [{"type": "text", "text": "Début"}, {"type": "image_url", "image_url": {"url": "retained:one"}},
              {"type": "text", "text": "🙂" * 4000}]
    head = api_photo_preview({"text": source, "api_turn_v1": {"history": "x" * 10000}})
    assert head == "Début\n" + "🙂" * 57
    with submission_turn_scope("a", "request", head):
        assert admission_submission(SimpleNamespace(message_id="other")) == {}
        metadata = admission_submission(SimpleNamespace(message_id="a"))
        assert metadata[PHOTO_PREVIEW_KEY] == head
    assert admission_submission(SimpleNamespace(message_id="a")) == {}
    store = backend.open(scope_a)
    store.sessions.create(NewSession("stamped", "api_server"))
    raw = f"Début\n\n{_hint('a' * 64)}\n[screenshot]\n" + "🙂" * 4000
    store.messages.append("stamped", NewMessage(role="user", content=raw, display_metadata=metadata))
    assert _preview(store, "stamped") == "Début " + "🙂" * 54 + "..."


def test_read_only_list_and_export_from_before_runtime_tables(tmp_path):
    from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
    from tests.fixtures.storage_holder import raw_holder
    from persistence.sqlite.conversation._store.sessions import list_sessions_rich

    raw = f"Souvenir\n\n{_hint('f' * 64)}\n[screenshot]"
    with open_persistence(profile_scope(home=tmp_path)) as source:
        source.sessions.create(NewSession("photo", "api_server"))
        source.messages.append("photo", NewMessage(role="user", content=raw, platform_message_id="old"))
        raw_holder(source)._execute_write(lambda conn: conn.execute("DROP TABLE session_admissions"))
    with open_persistence_reader(tmp_path / "state.db") as reader:
        assert list_sessions_rich(raw_holder(reader), limit=10)[0]["preview"] == "Souvenir"
        assert reader.maintenance.export_session("photo").messages[0].content == raw
