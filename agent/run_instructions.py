"""Apply authorized instructions to this agent's current model requests only."""
from contextlib import contextmanager
from contextvars import ContextVar

_active = ContextVar("model_run_instructions", default=None)


@contextmanager
def run_instruction_scope(agent, instructions):
    token = _active.set((agent, instructions) if instructions else None)
    try:
        yield
    finally:
        _active.reset(token)


def effective_system_prompt(agent, base):
    """Keep the reusable agent prompt intact, including for its background reviewers."""
    durable = getattr(agent, "ephemeral_system_prompt", None)
    active = _active.get()
    once = active[1] if active and active[0] is agent else None
    if not (durable or once):
        return base
    return "\n\n".join(part for part in (base, durable, once) if part).strip()


def valid_run_instructions(value):
    """JSON decoders can produce isolated surrogate code units which cannot reach a provider."""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False
