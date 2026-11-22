"""Preserve the provenance of historical API photo captions when admissions stay behind."""

from collections import defaultdict

from session_rules.photo_previews import PHOTO_PREVIEW_KEY, admitted_photo_preview, might_show_photo_receipt


def stamp_exported_photo_previews(holder, messages_by_session):
    if not holder.has_session_admissions:
        return
    pending = defaultdict(list)
    for session, messages in messages_by_session.items():
        for row in messages:
            owner = row.get("platform_message_id")
            content = row.get("content")
            metadata = row.get("display_metadata")
            if (owner and row.get("session_id") == session and row.get("role") == "user" and isinstance(content, str)
                    and (metadata is None or isinstance(metadata, dict) and PHOTO_PREVIEW_KEY not in metadata)
                    and might_show_photo_receipt(content)):
                pending[owner].append((session, row))
    owners = list(pending)
    for start in range(0, len(owners), 400):
        batch = owners[start:start + 400]
        marks = ",".join("?" for _ in batch)
        with holder._read_ctx() as conn:
            for owner, session, raw in conn.execute(
                    f"SELECT admission_id, target_session_id, payload_json FROM session_admissions WHERE admission_id IN ({marks})", batch):
                for expected, row in pending[owner]:
                    if session == expected and (head := admitted_photo_preview(raw)) is not None:
                        row["display_metadata"] = {**(row.get("display_metadata") or {}), PHOTO_PREVIEW_KEY: head}
