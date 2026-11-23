"""Platform scope rules for configured toolsets."""

from typing import Set


# Toolsets without a restriction entry are available on every platform.
_TOOLSET_PLATFORM_RESTRICTIONS = {
    "discord": {"discord"},
    "discord_admin": {"discord"},
    "mobile_ui": {"api_server"},
    "mobile_deliver": {"api_server", "cron"},
    "mobile_device": {"api_server"},
    "mobile_profiles": {"api_server"},
}


def toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return whether ``ts_key`` is available on ``platform``."""
    allowed: set[str] | None = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed
