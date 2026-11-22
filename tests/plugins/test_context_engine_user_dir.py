"""``context.engine: <name>`` names the active engine, so an engine dropped into
``$HERMES_HOME/plugins/<name>/`` loads without a ``plugins.enabled`` entry and never trips the
"Context engine 'X' not found — falling back to built-in compressor" warning (#61839)."""

import logging
import os
from pathlib import Path
from textwrap import dedent

import pytest

_ENGINE_SRC = dedent('''
    from agent.context_engine import ContextEngine

    class Demo(ContextEngine):
        @property
        def name(self):
            return "ctx_demo"
        def get_tool_schemas(self):
            return [{"type": "function", "function": {"name": "context_lookup"}}]
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages

    def register(ctx):
        ctx.register_command("context-lookup", lambda: None)
        ctx.register_context_engine(Demo())
''')


def test_user_installed_engine_is_selected_by_name(tmp_path: Path, monkeypatch, caplog):
    engine_dir = tmp_path / "plugins" / "ctx_demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(_ENGINE_SRC)
    (tmp_path / "plugins" / "notes").mkdir()  # unrelated user plugin: never treated as an engine
    (tmp_path / "plugins" / "notes" / "__init__.py").write_text("def register(ctx): pass\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.agent_init import _select_context_engine
    from plugins.context_engine import discover_context_engines, load_context_engine

    assert load_context_engine("notes") is None
    assert "ctx_demo" in {name for name, _, _ in discover_context_engines()}
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        engine = _select_context_engine({"context": {"engine": "ctx_demo"}})
    assert engine is not None and engine.name == "ctx_demo"
    assert "not found" not in caplog.text


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_schema_inspection_reads_profile_engine_without_registering_commands(tmp_path, monkeypatch, isolation):
    from hermes_cli.plugins import get_plugin_manager
    from plugins.plugin_loader import inspect_provider_tool_names

    config = {"context": {"engine": "schema_plugin"}, "plugins": {"isolation": isolation}}
    for profile in ("alpha", "beta"):
        home = tmp_path / profile
        engine_dir = home / "plugins" / "schema_plugin"
        engine_dir.mkdir(parents=True)
        tool_name = f"{profile}_lookup"
        source = _ENGINE_SRC.replace("context_lookup", tool_name)
        if isolation == "host":
            source = f"import os\nassert os.getpid() != {os.getpid()}\n" + source
        (engine_dir / "__init__.py").write_text(source)
        (home / "config.yaml").write_text(f"plugins:\n  isolation: {isolation}\n")
        monkeypatch.setenv("HERMES_HOME", str(home))

        manager = get_plugin_manager()
        assert inspect_provider_tool_names(config, "context_engine") == {tool_name}
        assert "context-lookup" not in manager._plugin_commands
        assert getattr(manager, "_plugin_host_instance", None) is None
        if isolation == "host":
            # A profile policy snapshot with an unavailable launcher must never use the ambient launcher.
            closed = {**config, "plugins": {"isolation": "host", "host": {"launcher": [str(home / "missing")]}}}
            assert inspect_provider_tool_names(closed, "context_engine") == set()


def test_schema_inspection_uses_general_engine_only_after_activation(tmp_path, monkeypatch):
    from hermes_cli.plugins import get_plugin_context_engine, get_plugin_manager
    from plugins.plugin_loader import inspect_provider_tool_names

    # The general plugin and its engine need not have the same name.
    plugin_dir = tmp_path / "plugins" / "general-pack"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(_ENGINE_SRC)
    (plugin_dir / "plugin.yaml").write_text("name: general-pack\n")
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [general-pack]\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = get_plugin_manager()
    config = {"context": {"engine": "ctx_demo"}}

    assert inspect_provider_tool_names(config, "context_engine") == set()
    assert "context-lookup" not in manager._plugin_commands
    engine = get_plugin_context_engine()
    assert engine.name == "ctx_demo"
    assert inspect_provider_tool_names(config, "context_engine") == {"context_lookup"}
    assert get_plugin_context_engine(discover=False) is engine
    config["plugins"] = {"isolation": "host"}
    assert inspect_provider_tool_names(config, "context_engine") == set()
