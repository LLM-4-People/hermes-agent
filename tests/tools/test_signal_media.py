"""Tests for Signal media delivery in send_message_tool.py."""

import asyncio
import base64
import sys
from types import ModuleType

import pytest


def _make_httpx_mock():
    """Create a mock httpx module with proper sync json()."""

    class AsyncBaseTransport:
        pass

    class Proxy:
        pass

    class MockResp:
        status_code = 200
        def json(self):
            return {"timestamp": 1234567890}
        def raise_for_status(self):
            pass

    class MockClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def post(self, *args, **kwargs):
            return MockResp()

    httpx_mock = ModuleType("httpx")
    httpx_mock.AsyncClient = lambda timeout=None: MockClient()
    httpx_mock.AsyncBaseTransport = AsyncBaseTransport  # Needed by Telegram adapter
    httpx_mock.Proxy = Proxy  # Needed by telegram-bot library
    return httpx_mock


@pytest.fixture(autouse=True)
def inject_httpx(monkeypatch):
    """Inject mock httpx into sys.modules before imports."""
    monkeypatch.setitem(sys.modules, "httpx", _make_httpx_mock())


class TestSendSignalMediaFiles:
    """Test that _send_signal correctly handles media_files parameter."""


    def test_send_signal_with_missing_media_file(self):
        """Missing media files should generate warnings but not fail."""
        from plugins.platforms.signal.standalone import _send_signal

        extra = {"http_url": "http://localhost:8080", "account": "+155****4567"}

        result = asyncio.run(
            _send_signal(extra, "+155****9999", "File missing?", media_files=[("/nonexistent.png", False)])
        )

        assert result["success"] is True  # Should succeed despite missing file
        assert result["warnings"]

    def test_send_signal_skips_an_oversize_path_upload(self, tmp_path):
        """Path mode must enforce the same 100 MB limit as inline mode."""
        from plugins.platforms.signal.standalone import _send_signal

        big = tmp_path / "big.png"
        with open(big, "wb") as f:
            f.truncate(101 * 1024 * 1024)  # sparse

        extra = {
            "http_url": "http://localhost:8080",
            "account": "+155****4567",
            "inline_attachments": "never",
        }

        result = asyncio.run(
            _send_signal(extra, "+155****9999", "too big?", media_files=[(str(big), False)]))

        assert result["success"] is True
        assert any("skipped" in warning for warning in result["warnings"])
        assert not any("not found" in warning for warning in result["warnings"])


class TestAttachmentDataUri:
    """The cross-container attachment representation owned by signal_media."""

    def test_encodes_mime_filename_and_bytes(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_data_uri

        image = tmp_path / "chart.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

        uri = asyncio.run(attachment_data_uri(image))

        assert uri.startswith("data:image/png;filename=chart.png;base64,")
        assert base64.b64decode(uri.split(";base64,", 1)[1]) == image.read_bytes()

    def test_rejects_an_oversized_file(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_data_uri

        image = tmp_path / "chart.png"
        image.write_bytes(b"\x89PNG" + b"\x00" * 8)

        with pytest.raises(ValueError, match="too large"):
            asyncio.run(attachment_data_uri(image, max_bytes=4))

    def test_percent_encodes_the_display_name(self, tmp_path):
        """signal-cli URL-decodes data-URI parameters, so delimiters and '%' must be escaped."""
        from plugins.platforms.signal.signal_media import attachment_data_uri

        image = tmp_path / "chart.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        uri = asyncio.run(attachment_data_uri(image, filename="100% done; v2.png"))

        assert "filename=100%25%20done%3B%20v2.png;" in uri

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, True), ("", True), ("always", True), ("ALWAYS", True), ("never", False), ("NEVER", False)],
    )
    def test_inline_mode_switch(self, value, expected):
        from plugins.platforms.signal.signal_media import inline_attachments_enabled

        assert inline_attachments_enabled(value) is expected

    @pytest.mark.parametrize("value", [False, 0, "false", "0", "no", "off"])
    def test_falsey_inline_mode_disables_inlining(self, value):
        """A YAML boolean or its string spelling must opt into path uploads, not fall back to always."""
        from plugins.platforms.signal.signal_media import inline_attachments_enabled

        assert inline_attachments_enabled(value) is False

    def test_mime_falls_back_to_the_path_extension_when_the_display_name_has_none(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_data_uri

        image = tmp_path / "chart.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        uri = asyncio.run(attachment_data_uri(image, filename="chart"))

        assert uri.startswith("data:image/png;filename=chart;base64,")

    def test_a_directory_is_reported_as_a_value_error(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_data_uri

        with pytest.raises(ValueError, match="not a regular file"):
            asyncio.run(attachment_data_uri(tmp_path))

    def test_a_read_failure_is_reported_as_a_value_error(self, tmp_path, monkeypatch):
        from pathlib import Path

        from plugins.platforms.signal.signal_media import attachment_data_uri_sync

        image = tmp_path / "chart.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")

        def denied(self):
            raise PermissionError("denied")

        monkeypatch.setattr(Path, "read_bytes", denied)
        with pytest.raises(ValueError, match="attachment unavailable: PermissionError"):
            attachment_data_uri_sync(image)

    def test_payload_size_counts_bytes_on_disk_and_serialized_data_uri_bytes(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_payload_size

        path = tmp_path / "blob.bin"
        path.write_bytes(b"x" * 42)
        payload = base64.b64encode(b"x" * 300).decode()
        encoded = "data:application/octet-stream;base64," + payload

        assert attachment_payload_size(str(path)) == 42
        assert attachment_payload_size(encoded) == len(payload)
        assert attachment_payload_size("/nonexistent") == 0

    def test_attachment_size_rejects_missing_directory_and_oversize(self, tmp_path):
        from plugins.platforms.signal.signal_media import attachment_size_sync

        with pytest.raises(ValueError, match="not a regular file"):
            attachment_size_sync(tmp_path / "missing.png")
        with pytest.raises(ValueError, match="not a regular file"):
            attachment_size_sync(tmp_path)

        path = tmp_path / "chart.png"
        path.write_bytes(b"x" * 8)
        assert attachment_size_sync(path) == 8
        with pytest.raises(ValueError, match="too large"):
            attachment_size_sync(path, max_bytes=4)
