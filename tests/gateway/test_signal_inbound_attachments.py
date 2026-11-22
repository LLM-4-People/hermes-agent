"""Inbound Signal attachments are fetched only for senders the gateway authorizes.

Drives the real adapter with the real GatewayRunner authorization callback wired as startup wires it;
only signal-cli's getAttachment RPC is stubbed.
"""
import base64

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig

OWNER, STRANGER = "+15550000001", "+15550000002"


@pytest.fixture
def signal(tmp_path, monkeypatch):
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.signal.adapter import SignalAdapter

    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "SIGNAL_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", OWNER)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path / "pairing")

    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"http_url": "http://127.0.0.1:1", "account": "+15559999999"}))
    rpc_calls = []

    async def rpc(method, params, rpc_id=None):
        rpc_calls.append(method)
        return {"data": base64.b64encode(b"%PDF-1.4 fake").decode()}

    adapter._rpc = rpc
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.SIGNAL: adapter.config})
    runner.adapters = {Platform.SIGNAL: adapter}
    runner.pairing_store = PairingStore()
    events = []

    async def capture(event):
        events.append(event)
    adapter.handle_message = capture

    async def dm(sender, text="see attached", attachment=None):
        before = len(events)
        att = attachment or {"id": "att1.pdf", "contentType": "application/pdf", "size": 13}
        await adapter._handle_envelope({"envelope": {
            "sourceNumber": sender, "sourceName": "someone", "timestamp": 1700000000000,
            "dataMessage": {"timestamp": 1700000000000, "message": text, "attachments": [att]}}})
        return events[before] if len(events) > before else None

    return adapter, runner, dm, rpc_calls


@pytest.mark.asyncio
async def test_attachments_are_fetched_only_for_authorized_senders(signal):
    adapter, runner, dm, rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    denied = await dm(STRANGER)
    assert rpc_calls == []
    assert denied.media_urls == []
    assert denied.source.user_id == STRANGER  # still dispatched so the runner can deny/pair
    assert await dm(STRANGER, text="") is None  # attachment-only from a stranger is contentless
    assert rpc_calls == []

    allowed = await dm(OWNER)
    assert rpc_calls == ["getAttachment"]
    assert len(allowed.media_urls) == 1


@pytest.mark.asyncio
async def test_unwired_authorization_check_fetches_nothing(signal):
    _adapter, _runner, dm, rpc_calls = signal

    event = await dm(OWNER)

    assert rpc_calls == []
    assert event.media_urls == []


@pytest.mark.asyncio
async def test_bot_users_sender_in_loop_guard_cooldown_gets_no_attachment_fetch(signal):
    """An allowlisted ``bot_users`` sender is judged as a bot: while its chat cools down, nothing is fetched."""
    from gateway.bot_loop_guard import BotLoopGuard
    adapter, runner, dm, rpc_calls = signal
    adapter.config.extra["bot_users"] = [OWNER]
    runner._bot_loop_guard = guard = BotLoopGuard()
    guard.blocked = lambda conversation: True
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    event = await dm(OWNER)

    assert event.source.is_bot
    assert rpc_calls == []
    assert event.media_urls == []


@pytest.mark.asyncio
async def test_inbound_document_keeps_the_sender_filename_and_extension(signal):
    """DOCX/XLSX/PPTX share the ZIP magic; the declared filename must win over ``.zip``."""
    from gateway.platforms.event import MessageType

    adapter, runner, dm, _rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    async def rpc(method, params, rpc_id=None):
        return {"data": base64.b64encode(b"PK\x03\x04fake-docx").decode()}

    adapter._rpc = rpc
    event = await dm(OWNER, attachment={
        "id": "att-docx",
        "contentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "filename": "Q3 plan.docx",
        "size": 12,
    })

    assert event.message_type == MessageType.DOCUMENT
    assert event.media_filenames == ["Q3 plan.docx"]
    assert event.media_urls[0].endswith("Q3 plan.docx")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declared_type",
    [
        "application/octet-stream",
        "application/octet-stream; charset=binary",
        "APPLICATION/OCTET-STREAM",
    ],
)
async def test_generic_content_type_does_not_mask_magic_byte_image_classification(signal, declared_type):
    """A sender-declared generic type must not turn a real PNG into a DOCUMENT."""
    from gateway.platforms.event import MessageType

    adapter, runner, dm, _rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    async def rpc(method, params, rpc_id=None):
        return {"data": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()}

    adapter._rpc = rpc
    event = await dm(OWNER, attachment={
        "id": "att-png", "contentType": declared_type, "size": 16,
    })

    assert event.message_type == MessageType.PHOTO
    assert event.media_types == ["image/png"]


@pytest.mark.asyncio
async def test_inbound_attachment_without_a_sender_filename_leaves_the_display_name_to_the_parser(signal):
    """Passing the cache basename would expose ``doc_<uuid>_``; the parser derives the suffix."""
    adapter, runner, dm, _rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    event = await dm(OWNER, attachment={"id": "att-pdf", "contentType": "application/pdf", "size": 13})

    assert event.media_filenames == [""]

    from gateway.run import GatewayRunner
    display, _ = GatewayRunner._inbound_attachment_display_name(
        "/tmp/cache/documents/doc_abc123_attachment.pdf", "")
    assert display == "attachment.pdf"


@pytest.mark.asyncio
async def test_decoded_attachment_over_the_limit_is_skipped(signal, monkeypatch):
    """The declared ``size`` is advisory; the decoded bytes are checked too."""
    adapter, runner, dm, _rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))
    monkeypatch.setattr("plugins.platforms.signal.adapter.SIGNAL_MAX_ATTACHMENT_SIZE", 4)

    event = await dm(OWNER, attachment={"id": "att-big", "contentType": "application/pdf"})

    assert event.media_urls == []
    assert event.media_types == []



@pytest.mark.asyncio
async def test_inbound_image_without_content_type_is_classified_from_bytes(signal):
    """signal-cli may omit contentType; the magic bytes still make this a PHOTO with an image MIME."""
    from gateway.platforms.event import MessageType

    adapter, runner, dm, _rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))

    async def rpc(method, params, rpc_id=None):
        return {"data": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()}

    adapter._rpc = rpc
    event = await dm(OWNER, attachment={"id": "att-png", "size": 16})

    assert event.message_type == MessageType.PHOTO
    assert event.media_types == ["image/png"]
    assert event.media_urls[0].endswith(".png")


@pytest.mark.asyncio
async def test_ignore_attachments_skips_the_fetch(signal):
    adapter, runner, dm, rpc_calls = signal
    adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.SIGNAL))
    adapter.ignore_attachments = True

    event = await dm(OWNER, text="")

    assert event is None  # attachment-only message becomes contentless
    assert rpc_calls == []
