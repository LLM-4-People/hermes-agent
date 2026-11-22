"""Apply authorized instructions to this agent's current model requests only."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

@dataclass
class _BoundInstructions:
    agent: object
    text: str


_active: ContextVar[_BoundInstructions | None] = ContextVar("model_run_instructions", default=None)


@contextmanager
def run_instruction_scope(agent, instructions):
    bound = _BoundInstructions(agent, instructions) if instructions else None
    token = _active.set(bound)
    try:
        yield
    finally:
        if bound is not None:
            # Copied execution contexts share this binding and may outlive the turn.
            bound.text = ""
            bound.agent = None
        _active.reset(token)


def effective_system_prompt(agent, base):
    """Keep the reusable agent prompt intact, including for its background reviewers."""
    durable = getattr(agent, "ephemeral_system_prompt", None)
    active = _active.get()
    once = active.text if active and active.agent is agent else None
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
