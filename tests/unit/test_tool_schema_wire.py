# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Structural pin for the LLM-facing tool schemas.

``schemas_as_provider_tools()`` emits the Anthropic-shape descriptors the model
sees for every tool in ``ALL_TOOLS``. Their STRUCTURE -- tool names, required
fields, property names and types (incl. nested ``$defs`` like ``EditPair``) --
is frozen LLM I/O: a silent drift changes what every model can call. This pins
that structure against a golden digest so a schema change is deliberate.

Description prose is EXCLUDED on purpose: it is tuned deliberately (small-model
phrasing) and would make the pin fight every wording tweak.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent6.tools.schema import (
    ASK_EXTRA_TOOLS,
    LOOP_EXTRA_TOOLS,
    MACHINE_EXTRA_TOOLS,
    PLAN_EXTRA_TOOLS,
    schemas_as_provider_tools,
)

_GOLDEN = Path(__file__).parent / "data" / "golden_tool_schemas.json"
# The loop-only control tools (finish_session, run_metric_command, the task
# tools, ask_user, use_skill), plan's finish_planning, ask's agent6_docs -- the
# LLM-facing surface OUTSIDE ALL_TOOLS that schemas_as_provider_tools() (and so
# _GOLDEN) does not cover. Pinned as a deduped, name-sorted digest.
_GOLDEN_EXTRA = Path(__file__).parent / "data" / "golden_extra_tool_schemas.json"


def _prop_type(sub: dict[str, Any]) -> str:
    """A property subschema -> a compact type descriptor (no description prose)."""
    if "type" in sub:
        t = sub["type"]
        if t == "array" and isinstance(sub.get("items"), dict):
            return f"array[{_prop_type(sub['items'])}]"
        return str(t)
    if "$ref" in sub:
        return sub["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in sub:
        return "anyOf[" + ",".join(sorted(_prop_type(s) for s in sub["anyOf"])) + "]"
    if "allOf" in sub:
        return "allOf[" + ",".join(_prop_type(s) for s in sub["allOf"]) + "]"
    return "enum" if "enum" in sub else "?"


def _props(schema: dict[str, Any]) -> dict[str, str]:
    return {k: _prop_type(v) for k, v in sorted(schema.get("properties", {}).items())}


def _entry(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": name,
        "required": sorted(schema.get("required", [])),
        "properties": _props(schema),
    }
    if schema.get("$defs"):
        entry["defs"] = {name: _props(d) for name, d in sorted(schema["$defs"].items())}
    return entry


def _digest() -> list[dict[str, Any]]:
    return [_entry(tool["name"], tool["input_schema"]) for tool in schemas_as_provider_tools()]


def _extra_digest() -> list[dict[str, Any]]:
    """Structural digest of every tool OUTSIDE ALL_TOOLS, deduped by name (dag_*
    appear in both LOOP and PLAN; finish_session in both LOOP and MACHINE) and sorted,
    so the pin is order-independent of the tuples."""
    by_name: dict[str, dict[str, Any]] = {}
    for cls in (*LOOP_EXTRA_TOOLS, *PLAN_EXTRA_TOOLS, *ASK_EXTRA_TOOLS, *MACHINE_EXTRA_TOOLS):
        schema = cls.model_json_schema()
        schema.setdefault("type", "object")
        by_name[cls.TOOL_NAME] = _entry(cls.TOOL_NAME, schema)
    return [by_name[name] for name in sorted(by_name)]


def test_tool_schemas_structure_matches_golden() -> None:
    generated = json.dumps(_digest(), indent=2) + "\n"
    committed = _GOLDEN.read_text(encoding="utf-8")
    assert generated == committed, (
        "LLM-facing tool schema structure drifted; if intended, regenerate the "
        'golden: python -c "import json,tests.unit.test_tool_schema_wire as t; '
        "open(t._GOLDEN,'w').write(json.dumps(t._digest(),indent=2)+chr(10))\""
    )


def test_extra_tool_schemas_structure_matches_golden() -> None:
    generated = json.dumps(_extra_digest(), indent=2) + "\n"
    committed = _GOLDEN_EXTRA.read_text(encoding="utf-8")
    assert generated == committed, (
        "loop/plan/ask/machine extra tool schema structure drifted; if intended, "
        'regenerate: python -c "import json,tests.unit.test_tool_schema_wire as t; '
        "open(t._GOLDEN_EXTRA,'w').write(json.dumps(t._extra_digest(),indent=2)+chr(10))\""
    )


def test_status_pattern_bytes_are_pinned() -> None:
    # The update_task/list_tasks status pattern is DERIVED from the NodeStatus
    # Literal (one owner); this pins the emitted LLM-facing bytes, so growing
    # the vocabulary is a deliberate schema change, not a silent one.
    from agent6.tools.schema import DagListTasksInput, DagUpdateTaskInput

    expected = "^(pending|in_progress|passed|failed|skipped|obsolete)$"
    # Both statuses are optional (update_task may carry only depends_on), so
    # the pattern sits on the string arm of an anyOf in each.
    for cls in (DagUpdateTaskInput, DagListTasksInput):
        anyof = cls.model_json_schema()["properties"]["status"]["anyOf"]
        assert [s.get("pattern") for s in anyof if s.get("type") == "string"] == [expected], cls


def test_an_ask_user_question_is_bounded_like_every_other_model_string() -> None:
    """The question and its options reach the journal, an ACP permission title,
    the TUI modal and the web composer verbatim. Uncapped, a model that ran
    away wrote all of them: every sibling string in this schema is bounded, and
    this one was the exception."""
    import pytest
    from pydantic import ValidationError

    from agent6.tools.schema import AskUserInput

    ok = AskUserInput.model_validate(
        {"questions": [{"question": "which port?", "options": ["80"]}]}
    )
    assert ok.questions[0].question == "which port?"

    with pytest.raises(ValidationError, match="at most 2000 characters"):
        AskUserInput.model_validate({"questions": [{"question": "x" * 2_001}]})
    with pytest.raises(ValidationError, match="at most 200 characters"):
        AskUserInput.model_validate({"questions": [{"question": "q", "options": ["y" * 201]}]})


def test_add_task_parent_id_carries_the_ulid_constraint() -> None:
    """parent_id was the one task-ULID param without the 26-char rule its
    siblings enforce; "" passed the schema and silently attached the task to
    the run root. None still means root; a malformed id fails loud."""
    import pytest
    from pydantic import ValidationError

    from agent6.tools.schema import DagAddTaskInput

    DagAddTaskInput(title="t")  # omitted -> root, unchanged
    DagAddTaskInput(title="t", parent_id="01ARZ3NDEKTSV4RRFFQ69G5FAV")
    with pytest.raises(ValidationError):
        DagAddTaskInput(title="t", parent_id="")
    with pytest.raises(ValidationError):
        DagAddTaskInput(title="t", parent_id="short")


def test_wire_schema_strips_schema_titles_but_keeps_a_field_named_title() -> None:
    """The loop's tool list and the descriptor dump share one schema builder;
    it drops pydantic's schema-level "title" noise and nothing else. The old
    stripper dropped every "title" key, add_task's `title` FIELD included (the
    descriptor dump alone had it, so tests pinned a schema the model never
    saw and the model would have seen add_task without its one required
    field once the loop shared it)."""
    from agent6.tools.schema import DagAddTaskInput, ReadFileInput, wire_schema
    from agent6.workflows._toolset import tool_definitions

    add_task = wire_schema(DagAddTaskInput)
    assert "title" in add_task["properties"] and add_task["required"] == ["title"]
    assert "title" not in add_task and "title" not in add_task["properties"]["title"]
    read_file = wire_schema(ReadFileInput)
    assert "title" not in read_file["properties"]["path"]

    class _Dispatcher:
        def available_tool_names(self) -> tuple[str, ...]:
            return ("read_file",)

        def metric_configured(self) -> bool:
            return True

        def tool_is_withheld(self, _name: str) -> bool:
            return False

        def skills_available(self) -> bool:
            return False

        def mcp_descriptors(self) -> tuple[object, ...]:
            return ()

    definitions = tool_definitions(_Dispatcher(), mode="run")  # type: ignore[arg-type]
    by_name = {d.name: d for d in definitions}
    assert by_name["read_file"].input_schema == read_file
    assert by_name["add_task"].input_schema == add_task
