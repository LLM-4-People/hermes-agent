"""Tool execution restrictions for hosted scopes and per-run tool selections.

A client may narrow the operator's toolsets for its run. The effective tool names follow the
execution context into child agents and workers. The model's advertised schemas and executor
both consult the selection. Provider-injected tools additionally require their own agent handler.

In a hosted scope nothing tenant-supplied may execute on this host or touch its filesystem, and no
agent may read or write state this host keeps for its profile. Two decisions enforce that. Each is
made in one function and is deny-by-default, so a new tool or feature stays off in a hosted scope
until someone shows it acts only through the agent's sandbox:

* :func:`hosted_tool_refusal` — a tool runs in a hosted scope only if it is on
  :data:`HOSTED_TOOL_ALLOWLIST`, the tools whose every effect goes through the agent's own sandbox
  (``tools.environments.sandbox_scope``). Everything else (memory, recall, todo, vision, browser, web,
  MCP, connectors, catalog, delegation, skills, cron, ...) is refused before its handler. The agent
  loop asks once per call, in ``agent.tool_executor._dispatch_authorized_once``: the one function every
  sequential, concurrent and batched call goes through, whatever executes it (inline agent tools,
  memory-provider and context-engine tools, the registry, connectors). The execution primitives also
  ask, for callers outside the loop: ``ToolRegistry.dispatch`` (execute_code RPC, the session kernel,
  the MCP bridge, plugin ``dispatch_tool``), the inline-executor table and connector dispatch.
* :func:`host_feature_enabled` — the agent features outside tool dispatch that read or write this
  host's state for the profile, or run here (:data:`HOST_FEATURES`). Every one is off in a hosted
  scope; each site asks this one function. They come back per agent through storage ports.

A local install uses its normal tool policy unless the run supplies a selection.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

# Tools whose every effect goes through the agent's ScopedEnvironment and nothing on this host:
#   terminal        commands run by env.execute in the sandbox; background spawns via spawn_via_env,
#                   their records stamped with the sandbox key.
#   process_manage  acts only on the bound agent's own records (sandbox_visible); polls via its env.
#   execute_code    always the remote kernel / per-call script in the sandbox, never the host kernel;
#                   its RPC tool calls re-enter ToolRegistry.dispatch, so they meet this allowlist too.
#   read_file, write_file, patch, search_files
#                   ShellFileOperations over the sandbox env: no host fast path, a lexical path policy,
#                   no host document extraction; an oversized result spills into the sandbox.
HOSTED_TOOL_ALLOWLIST = frozenset({
    "terminal", "process_manage", "execute_code", "read_file", "write_file", "patch", "search_files",
})

# Agent features outside tool dispatch that read or write this host's profile state, or run here.
HOST_FEATURES = frozenset({
    "context_files",              # AGENTS.md / CLAUDE.md / .cursorrules / SOUL.md in the system prompt
    "subdirectory_hints",         # AGENTS.md found beside the paths a tool call names
    "context_references",         # @file / @diff / @url expansion of the user's message
    "workspace_snapshot",         # git and project facts for the system prompt
    "backend_probe",              # a throwaway environment built to describe the backend
    "builtin_memory",             # MEMORY.md / USER.md and external memory providers
    "session_recall",             # the profile's state.db opened for recall
    "background_review",          # the post-turn memory/skill review fork
    "trajectory_export",          # trajectory JSONL in the working directory
    "file_mutation_bookkeeping",  # host stat of a failed write's target
    "checkpoints",                # shadow-git checkpoints and rollback of the working directory
    "media_processing",           # images/video decoded, converted and cached on this host
    "skills",                     # the profile's skill files: index, preload, auto-load, slash commands
    "skill_inline_shell",         # !`cmd` blocks in a skill body
    "cron_scripts",               # a cron job's pre-run script
    "shell_hooks",                # operator shell hooks: a host process fed the event payload
    "tool_host_artifacts",        # files an allowed tool keeps here about a call: terminal output
                                  # spill, saved blocked scripts, the verification ledger
    "mcp_stdio",                  # a profile MCP server started as a host process
    "media_delivery",             # a reply attachment named by host path
    "transcription",              # audio read and speech engines run here
    "delegation",                 # child agents started outside a tool call (/review)
})


_RUN_TOOLS = ContextVar("run_tools", default=None)


@contextmanager
def restrict_tools(toolsets, names, exclusions):
    parent = _RUN_TOOLS.get()
    chosen, allowed = frozenset(toolsets), frozenset(names)
    excluded = frozenset(exclusions)
    if parent is not None:
        excluded |= parent[2]
        chosen, allowed = chosen & parent[0], allowed & parent[1]
    token = _RUN_TOOLS.set((chosen, allowed, excluded))
    try:
        yield
    finally:
        _RUN_TOOLS.reset(token)


def scoped_toolsets(toolsets):
    selection = _RUN_TOOLS.get()
    if selection is None:
        return toolsets
    return sorted(selection[0] if toolsets is None else selection[0].intersection(toolsets))


def run_tool_names():
    selection = _RUN_TOOLS.get()
    return selection[1] if selection is not None else None


def run_policy_key():
    selection = _RUN_TOOLS.get()
    return [sorted(part) for part in selection] if selection is not None else None


def _run_permits(name, selection, agent):
    toolsets, names, excluded = selection
    if name in excluded:
        return False
    if name in names:
        return True
    if name == "request_device_action" and "mobile_device" in toolsets:
        from tools.mobile_device_bridge import current_device_capabilities
        return bool(current_device_capabilities())
    if agent is None or name not in getattr(agent, "valid_tool_names", ()):
        return False
    # These two providers attach tools directly to an agent after registry discovery. A grant
    # follows only that agent's real handler. Registry and inline dispatch keep their own fence.
    from agent.inline_tool_executors import is_context_engine_tool, is_memory_provider_tool
    return (("memory" in toolsets and is_memory_provider_tool(agent, name))
            or ("context_engine" in toolsets and is_context_engine_tool(agent, name)))


def scoped_tool_schemas(agent, schemas):
    selection = _RUN_TOOLS.get()
    if selection is None or not schemas:
        return schemas
    return [schema for schema in schemas if _run_permits(schema["function"]["name"], selection, agent)]


def run_tool_refusal(name: str, agent=None) -> Optional[str]:
    selection = _RUN_TOOLS.get()
    if selection is not None and not _run_permits(name, selection, agent):
        return f"The tool {name!r} is outside this run's selected toolsets."
    return None


def tool_execution_refusal(name: str, agent=None) -> Optional[str]:
    return hosted_tool_refusal(name) or run_tool_refusal(name, agent)


class HostedToolRefused(PermissionError):
    """A hosted scope asked for something that would run or read on this host."""


def _hosted() -> bool:
    from tools.environments.sandbox_scope import hosted_scope
    return hosted_scope() is not None


def _message(what: str, why: str) -> str:
    return (f"{what} is not available to a hosted agent: {why}. A hosted agent's tools are "
            f"{', '.join(sorted(HOSTED_TOOL_ALLOWLIST))}, all run in its own sandbox.")


def hosted_tool_refusal(tool_name: str) -> Optional[str]:
    """Why a hosted scope may not run *tool_name*, or ``None``. Always ``None`` outside a hosted scope,
    where the allowlist is not consulted."""
    if not _hosted() or tool_name in HOSTED_TOOL_ALLOWLIST:
        return None
    return _message(tool_name, "only tools that act solely through the agent's sandbox are enabled")


def host_feature_enabled(feature: str) -> bool:
    """Whether *feature* (one of :data:`HOST_FEATURES`) may use this host's state now: always outside
    a hosted scope, never inside one."""
    if feature not in HOST_FEATURES:
        raise KeyError(f"unknown host feature {feature!r}")
    return not _hosted()


def host_feature_refusal(feature: str, what: str) -> Optional[str]:
    """:func:`host_feature_enabled` as a message for *what*: the refusal in a hosted scope, else ``None``."""
    if host_feature_enabled(feature):
        return None
    return _message(what, f"it uses this host's {feature.replace('_', ' ')}")


def require_host_feature(feature: str, what: str) -> None:
    """Raise :class:`HostedToolRefused` when *feature* is off (a hosted scope); a no-op otherwise."""
    if refusal := host_feature_refusal(feature, what):
        raise HostedToolRefused(refusal)
