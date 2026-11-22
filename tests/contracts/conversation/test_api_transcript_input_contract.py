"""A visible input receipt is proved by the admission ledger of this profile only."""
from persistence.conversation.messages import TranscriptRead
from tests.fixtures.storage_seed import seed_admission, seed_messages, seed_session


def test_input_identity_does_not_follow_copied_messages_across_profiles(backend, scope_a, scope_b):
    a, b = backend.open(scope_a), backend.open(scope_b)
    sid = "same-thread-id"
    for store in (a, b):
        seed_session(store, sid, "api_server")
    marker = seed_admission(a, sid, "first", "terminal", "completed", principal_id="api")
    message = {"role": "user", "content": "My guide", "platform_message_id": marker}
    for store in (a, b):
        seed_messages(store, sid, message)
    assert a.messages.read_transcript_page(sid, TranscriptRead(), visible_only=True).rows[0].api_admission_id == marker
    assert b.messages.read_transcript_page(sid, TranscriptRead(), visible_only=True).rows[0].api_admission_id is None
    assert a.messages.read_transcript_page(sid, TranscriptRead()).rows[0].api_admission_id is None

    # A repeated marker cannot identify either row, even when its twin is beyond the returned page.
    seed_messages(a, sid, message)
    page = a.messages.read_transcript_page(sid, TranscriptRead(limit=1), visible_only=True)
    assert page.total == 2 and page.rows[0].api_admission_id is None
