"""Skill Studio CRUD API routes for ``APIServerAdapter``.

Reuses ``tools.skills_catalog_ops`` and ``tools.skill_manager_tool`` as the
single source of truth for discovery, validation, locking, linting, and cache
invalidation.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from tools.skill_manager_tool import (
    _create_skill,
    _delete_skill,
    _edit_skill,
    _patch_skill,
    _skill_mutation_lock,
    _validate_category,
    _validate_name,
)
from tools.skills_catalog_ops import (
    invalidate_skills_caches,
    is_skill_writable,
    list_skills_enriched,
    read_skill_detail,
    toggle_skill_enabled,
    validate_skill_locator,
)

logger = logging.getLogger("gateway.platforms.api_server")

CAPABILITY_FEATURES: dict[str, Any] = {
    "skill_studio": True,
}

CAPABILITY_ENDPOINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("skill_detail", ("GET", "/v1/skills/{name}", "skill_studio")),
    ("skill_create", ("POST", "/v1/skills", "skill_studio")),
    ("skill_update", ("PUT", "/v1/skills/{name}", "skill_studio")),
    ("skill_toggle", ("PUT", "/v1/skills/{name}/enabled", "skill_studio")),
    ("skill_delete", ("DELETE", "/v1/skills/{name}", "skill_studio")),
)


def _validate_utf8(value: Any, field: str) -> Optional[str]:
    if not isinstance(value, str):
        return f"'{field}' must be a string."
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return f"'{field}' must be valid UTF-8 text."
    return None


async def handle_list_skills(adapter, request: web.Request) -> web.Response:
    """GET /v1/skills — enriched skill catalog (supports ``?include_disabled=true``)."""
    from gateway.platforms.api_server import _coerce_request_bool, _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    include_disabled = _coerce_request_bool(request.query.get("include_disabled"), default=False)
    category = (request.query.get("category") or "").strip() or None
    tag = (request.query.get("tag") or "").strip() or None
    try:
        skills = await asyncio.to_thread(
            list_skills_enriched,
            include_disabled=include_disabled,
            platform="api_server",
            category=category,
            tag=tag,
        )
    except Exception:
        logger.exception("[%s] GET /v1/skills failed", adapter.name)
        return _error_response("Failed to enumerate skills.", 500, err_type="server_error", code="skills_list_failed")
    return web.json_response({"object": "list", "data": skills})


async def handle_get_skill(adapter, request: web.Request) -> web.Response:
    """GET /v1/skills/{name:.+} — full SKILL.md read with frontmatter, body, and lint."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    name = request.match_info["name"]
    detail, err = await asyncio.to_thread(read_skill_detail, name, platform="api_server")
    if err is not None:
        status, code, msg = err
        return _error_response(msg, status, code=code)
    assert detail is not None
    return web.json_response({"object": "hermes.skill", **detail})


def _sync_create_skill(name: str, content: str, category: Optional[str]) -> tuple[int, dict[str, Any]]:
    if err := (_validate_name(name) or _validate_category(category)):
        return 400, {"error": err, "code": "invalid_skill"}
    with _skill_mutation_lock(name):
        result = _create_skill(name, content, category)
    if not result.get("success"):
        msg = str(result.get("error") or "Failed to create skill.")
        status = 409 if "already exists" in msg.lower() else 400
        code = "skill_exists" if status == 409 else "invalid_skill"
        return status, {"error": msg, "code": code}
    invalidate_skills_caches()
    detail, _ = read_skill_detail(name, platform="api_server")
    if detail is None and category:
        detail, _ = read_skill_detail(f"{category}/{name}", platform="api_server")
    return 201, {
        "object": "hermes.skill",
        "skill": detail or {"name": name, "category": category},
        "message": result.get("message", f"Skill '{name}' created."),
        "lint_warnings": result.get("lint_warnings", []),
    }


async def handle_create_skill(adapter, request: web.Request) -> web.Response:
    """POST /v1/skills — create a user-owned skill in the active profile."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    name = body.get("name")
    content = body.get("content")
    category = body.get("category")
    for field, value in (("name", name), ("content", content)):
        if text_err := _validate_utf8(value, field):
            return _error_response(text_err, 400, code="invalid_skill")
    if category is not None:
        if text_err := _validate_utf8(category, "category"):
            return _error_response(text_err, 400, code="invalid_skill")
        category = category.strip() or None
    status, payload = await asyncio.to_thread(_sync_create_skill, name.strip(), content, category)
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload, status=status)


def _sync_update_skill(
    name: str,
    *,
    content: Optional[str],
    old_string: Optional[str],
    new_string: Optional[str],
    replace_all: bool,
) -> tuple[int, dict[str, Any]]:
    writable, err = is_skill_writable(name, platform="api_server")
    if not writable and err is not None:
        return err[0], {"error": err[2], "code": err[1]}
    with _skill_mutation_lock(name):
        if content is not None:
            result = _edit_skill(name, content)
        else:
            assert old_string is not None and new_string is not None
            result = _patch_skill(name, old_string, new_string, replace_all=replace_all)
    if not result.get("success"):
        msg = str(result.get("error") or "Failed to update skill.")
        status = 404 if "not found" in msg.lower() else 400
        return status, {"error": msg, "code": "skill_update_failed"}
    invalidate_skills_caches()
    detail, _ = read_skill_detail(name, platform="api_server")
    return 200, {
        "object": "hermes.skill",
        "skill": detail or {"name": name},
        "message": result.get("message", f"Skill '{name}' updated."),
        "lint_warnings": result.get("lint_warnings", (detail or {}).get("lint_warnings", [])),
    }


async def handle_update_skill(adapter, request: web.Request) -> web.Response:
    """PUT /v1/skills/{name:.+} — full rewrite (``content``) or targeted patch."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    name = request.match_info["name"]
    if loc_err := validate_skill_locator(name):
        return _error_response(loc_err, 400, code="invalid_skill_name")
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err

    has_content = "content" in body
    has_patch = "old_string" in body or "new_string" in body
    if has_content == has_patch:
        return _error_response(
            "Provide either 'content' (full rewrite) or 'old_string'+'new_string' (patch).",
            400,
            code="invalid_skill_update",
        )
    content = body.get("content") if has_content else None
    old_string = body.get("old_string") if has_patch else None
    new_string = body.get("new_string") if has_patch else None
    replace_all = body.get("replace_all", False)
    if not isinstance(replace_all, bool):
        return _error_response("'replace_all' must be a boolean.", 400, code="invalid_skill_update")
    if has_content:
        if text_err := _validate_utf8(content, "content"):
            return _error_response(text_err, 400, code="invalid_skill_update")
    else:
        for field, value in (("old_string", old_string), ("new_string", new_string)):
            if text_err := _validate_utf8(value, field):
                return _error_response(text_err, 400, code="invalid_skill_update")

    status, payload = await asyncio.to_thread(
        _sync_update_skill,
        name,
        content=content,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload)


async def handle_toggle_skill(adapter, request: web.Request) -> web.Response:
    """PUT /v1/skills/{name:.+}/enabled — toggle skill enabled state in profile config."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    name = request.match_info["name"]
    body, err = await adapter._read_json_body(request)
    if err is not None:
        return err
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        return _error_response("'enabled' must be a boolean.", 400, code="invalid_enabled_flag")
    platform = body.get("platform", "global")
    if not isinstance(platform, str):
        return _error_response("'platform' must be 'global' or 'api_server'.", 400, code="invalid_platform")
    result, toggle_err = await asyncio.to_thread(toggle_skill_enabled, name, enabled, platform=platform.strip())
    if toggle_err is not None:
        status, code, msg = toggle_err
        return _error_response(msg, status, code=code)
    assert result is not None
    return web.json_response({"object": "hermes.skill.toggle", "ok": True, **result})


def _sync_delete_skill(name: str) -> tuple[int, dict[str, Any]]:
    writable, err = is_skill_writable(name, platform="api_server")
    if not writable and err is not None:
        return err[0], {"error": err[2], "code": err[1]}
    with _skill_mutation_lock(name):
        result = _delete_skill(name)
    if not result.get("success"):
        msg = str(result.get("error") or "Failed to delete skill.")
        status = 404 if "not found" in msg.lower() else 400
        return status, {"error": msg, "code": "skill_delete_failed"}
    invalidate_skills_caches()
    return 200, {"object": "hermes.skill.deleted", "name": name, "deleted": True}


async def handle_delete_skill(adapter, request: web.Request) -> web.Response:
    """DELETE /v1/skills/{name:.+} — delete a user-owned writable skill."""
    from gateway.platforms.api_server import _error_response

    if auth_err := adapter._check_auth(request):
        return auth_err
    name = request.match_info["name"]
    if loc_err := validate_skill_locator(name):
        return _error_response(loc_err, 400, code="invalid_skill_name")
    status, payload = await asyncio.to_thread(_sync_delete_skill, name)
    if status >= 400:
        return _error_response(payload["error"], status, code=payload.get("code"))
    return web.json_response(payload)


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/v1/skills", functools.partial(handle_list_skills, adapter)),
        ("POST", "/v1/skills", functools.partial(handle_create_skill, adapter)),
        ("PUT", "/v1/skills/{name:.+}/enabled", functools.partial(handle_toggle_skill, adapter)),
        ("GET", "/v1/skills/{name:.+}", functools.partial(handle_get_skill, adapter)),
        ("PUT", "/v1/skills/{name:.+}", functools.partial(handle_update_skill, adapter)),
        ("DELETE", "/v1/skills/{name:.+}", functools.partial(handle_delete_skill, adapter)),
    ]
