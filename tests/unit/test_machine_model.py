# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.machine.model — `.asm.toml` parse + semantic validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.machine._semantics import load_machine
from agent6.machine.model import AgentState, MachineError

# The worked example from STATE_MACHINES.md §10. The canonical
# happy path; error-case tests mutate a copy of this.
VALID_MACHINE = """
machine = "item-classifier"
version = 1
initial = "poll"

[budget]
max_usd         = 25.0
max_transitions = 100000

[vars.operator]
inbox_dir = { type = "str", value = "/srv/inbox" }
poll_secs = { type = "int", value = 300 }

[vars.code]
pending = { type = "list[str]", default = [] }
cursor  = { type = "str",       default = "" }

[vars.agent]
verdict = { type = "classification", default = {} }

[schemas.classification]
label      = { type = "str", enum = ["urgent", "normal", "spam"] }
confidence = "float"

[schemas.scan_result]
pending = "list[str]"
cursor  = "str"

[states.poll]
kind = "wait"
every_secs = "{{ poll_secs }}"
on = { tick = "scan", signal = "scan" }

[states.scan]
kind = "tool"
command = ["scan-inbox", "--dir", "{{ inbox_dir }}", "--since", "{{ cursor }}"]
output_schema = "scan_result"
capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }
timeout_secs = 60
on = { ok = "have_items", nonzero = "poll", timeout = "poll" }

[states.have_items]
kind = "branch"
when = [
  { if = "len(pending) == 0", goto = "poll" },
  { else = true,              goto = "classify" },
]

[states.classify]
kind  = "agent"
model = "claude-sonnet-4-5"
prompt = "Classify these pending items: {{ pending | json }}"
output_schema = "classification"
capture = { finish_json = "verdict" }
timeout_secs = 600
on = { ok = "route", failed = "poll", budget_exhausted = "halt", timeout = "poll" }

[states.route]
kind = "branch"
when = [
  { if = "verdict.label == 'urgent' and verdict.confidence >= 0.7", goto = "record" },
  { else = true, goto = "poll" },
]

[states.record]
kind = "tool"
command = ["archive-item", "--label", "{{ verdict.label }}", "{{ pending }}"]
timeout_secs = 30
on = { ok = "poll", nonzero = "poll", timeout = "poll" }

[states.halt]
kind   = "terminal"
status = "failed"
reason = "machine budget exhausted"
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "m.asm.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _problems(tmp_path: Path, body: str) -> list[str]:
    with pytest.raises(MachineError) as excinfo:
        load_machine(_write(tmp_path, body))
    return excinfo.value.problems


def test_valid_machine_loads(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, VALID_MACHINE))
    assert spec.machine == "item-classifier"
    assert spec.initial == "poll"
    assert set(spec.states) == {
        "poll",
        "scan",
        "have_items",
        "classify",
        "route",
        "record",
        "halt",
    }


def test_agent_state_model_defaults_to_inherit(tmp_path: Path) -> None:
    # Omitting `model` on an agent state is valid and defaults to "inherit"
    # (the operator's worker model) — so an LLM-authored machine need not
    # hardcode a model the operator may not have configured.
    body = VALID_MACHINE.replace('\nmodel = "claude-sonnet-4-5"', "")
    spec = load_machine(_write(tmp_path, body))
    classify = spec.states["classify"]
    assert isinstance(classify, AgentState)
    assert classify.model == "inherit"


def test_bad_toml(tmp_path: Path) -> None:
    problems = _problems(tmp_path, "machine = ")
    assert any("not valid TOML" in p for p in problems)


def test_non_utf8_file_raises_machine_error(tmp_path: Path) -> None:
    # A non-UTF-8 .asm.toml must surface as a MachineError (which the CLI catches
    # and prints cleanly), not an unhandled UnicodeDecodeError that crashes
    # through the generic handler.
    path = tmp_path / "m.asm.toml"
    path.write_bytes(b"machine = \xff\xfe not utf-8")
    with pytest.raises(MachineError) as excinfo:
        load_machine(path)
    assert any("UTF-8" in p for p in excinfo.value.problems)


# -- naming rules ----------------------------------------------------------


def test_duplicate_name_across_owners(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[vars.agent]\nverdict = { type = "classification", default = {} }',
        '[vars.agent]\nverdict = { type = "classification", default = {} }\n'
        'cursor = { type = "str", default = "" }',
    )
    problems = _problems(tmp_path, body)
    assert any("declared in both" in p and "cursor" in p for p in problems)


def test_bare_top_level_var(tmp_path: Path) -> None:
    body = VALID_MACHINE + '\n[vars.stray]\nx = { type = "str", default = "" }\n'
    # `vars.stray` becomes an owner-less subtable.
    problems = _problems(tmp_path, body)
    assert any("no owner subtable" in p for p in problems)


def test_reserved_name(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        "[vars.code]\npending",
        '[vars.code]\nresult = { type = "str", default = "" }\npending',
    )
    problems = _problems(tmp_path, body)
    assert any("reserved" in p and "result" in p for p in problems)


def test_cron_wait_is_an_unknown_key(tmp_path: Path) -> None:
    # `wait` timings are every_secs and until; a `cron` key refuses at load
    # like any other unknown key (extra = "forbid"), named in the problem.
    body = VALID_MACHINE.replace('every_secs = "{{ poll_secs }}"', 'cron = "0 * * * *"')
    problems = _problems(tmp_path, body)
    assert any("cron" in p for p in problems)


def test_non_identifier_variable(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'cursor  = { type = "str",       default = "" }',
        'cursor  = { type = "str",       default = "" }\n'
        '"last-seen" = { type = "str", default = "" }',
    )
    problems = _problems(tmp_path, body)
    assert any("not a valid identifier" in p for p in problems)


# -- ownership wall --------------------------------------------------------


def test_tool_cannot_write_agent_var(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { verdict = "{{ result.pending }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any("may only write `[vars.code]`" in p for p in problems)


def test_undeclared_capture_target_names_where_to_declare_it(tmp_path: Path) -> None:
    """The diagnostic states the accepted form, not just the miss.

    A create attempt burned on 'is not a declared variable' left the model
    guessing where a declaration goes; the message now names [vars.<owner>].
    """
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { nonesuch = "{{ result.pending }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any("declare it in [vars.code]" in p for p in problems)


def test_capture_type_mismatch_names_the_type_to_declare(tmp_path: Path) -> None:
    """The type-mismatch diagnostic states the declaration that would fit."""
    body = VALID_MACHINE.replace(
        'verdict = { type = "classification", default = {} }',
        'verdict = { type = "json", default = {} }',
    )
    problems = _problems(tmp_path, body)
    assert any('declare it as type = "classification"' in p for p in problems)


def test_set_assignment_type_mismatch_names_the_type_to_declare(tmp_path: Path) -> None:
    """The lone-ref set mismatch states the declaration that would fit."""
    body = VALID_MACHINE.replace(
        'pending = { type = "list[str]", default = [] }',
        'pending = { type = "int", default = 0 }',
    )
    problems = _problems(tmp_path, body)
    assert any('declare it as type = "list[str]"' in p for p in problems)


def test_template_set_into_non_str_names_the_str_declaration(tmp_path: Path) -> None:
    """The rendered-template mismatch states the str declaration and the lone-ref out."""
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { pending = "c={{ result.cursor }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any(
        'declare it as type = "str" (only a lone {{ var }} keeps a value\'s type)' in p
        for p in problems
    )


def test_capture_cannot_write_operator_var(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'capture = { set = { pending = "{{ result.pending }}", cursor = "{{ result.cursor }}" } }',
        'capture = { set = { poll_secs = "{{ result.cursor }}" } }',
    )
    problems = _problems(tmp_path, body)
    assert any("owned by `[vars.operator]`" in p for p in problems)


# -- branches --------------------------------------------------------------


def test_branch_not_total(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '  { if = "len(pending) == 0", goto = "poll" },\n'
        '  { else = true,              goto = "classify" },',
        '  { if = "len(pending) == 0", goto = "poll" },',
    )
    problems = _problems(tmp_path, body)
    assert any("not total" in p for p in problems)


def test_branch_else_must_be_last(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '  { if = "len(pending) == 0", goto = "poll" },\n'
        '  { else = true,              goto = "classify" },',
        '  { else = true, goto = "poll" },\n  { if = "len(pending) == 0", goto = "classify" },',
    )
    problems = _problems(tmp_path, body)
    assert any("must be the final" in p for p in problems)


def test_predicate_misspelled_field(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("verdict.confidence >= 0.7", "verdict.confidense >= 0.7")
    problems = _problems(tmp_path, body)
    assert any("has no field" in p and "confidense" in p for p in problems)


def test_predicate_unknown_variable(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("len(pending) == 0", "len(nonsense) == 0")
    problems = _problems(tmp_path, body)
    assert any("unknown variable" in p and "nonsense" in p for p in problems)


def test_predicate_toml_boolean_literal_hints_python_form(tmp_path: Path) -> None:
    # `flag == true` reads `true` as an undeclared name; the error should point at
    # the Python literal rather than a bare "unknown variable".
    body = VALID_MACHINE.replace("len(pending) == 0", "len(pending) == 0 and pending == true")
    problems = _problems(tmp_path, body)
    assert any("True/False/None" in p for p in problems)


def test_predicate_len_of_int_rejected_at_load(tmp_path: Path) -> None:
    # `len(poll_secs)` (poll_secs is int) is a guaranteed runtime PredicateError;
    # it must be caught at load, mirroring the template `| len` filter check.
    body = VALID_MACHINE.replace("len(pending) == 0", "len(poll_secs) == 0")
    problems = _problems(tmp_path, body)
    assert any("`len()` does not apply to int" in p and "poll_secs" in p for p in problems)


def test_template_len_of_int_rejected_at_load(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("{{ pending | json }}", "{{ poll_secs | len }}")
    problems = _problems(tmp_path, body)
    assert any("`| len` does not apply to int" in p and "poll_secs" in p for p in problems)


def test_template_len_of_list_allowed(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace("{{ pending | json }}", "{{ pending | len }}")
    load_machine(_write(tmp_path, body))


def test_predicate_len_of_str_allowed(tmp_path: Path) -> None:
    # `len(cursor)` (cursor is str) is fine and must NOT be flagged.
    body = VALID_MACHINE.replace("len(pending) == 0", "len(cursor) == 0")
    # No MachineError raised means the machine validated cleanly.
    load_machine(_write(tmp_path, body))


def test_wait_every_secs_float_ref_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'poll_secs = { type = "int", value = 300 }',
        'poll_secs = { type = "float", value = 300.0 }',
    )
    problems = _problems(tmp_path, body)
    assert any("every_secs" in p and "int variable" in p for p in problems)


def test_wait_every_secs_zero_literal_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('every_secs = "{{ poll_secs }}"', 'every_secs = "0"')
    problems = _problems(tmp_path, body)
    assert any("every_secs" in p and ">= 1" in p for p in problems)


def test_wait_every_secs_zero_operator_ref_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'poll_secs = { type = "int", value = 300 }',
        'poll_secs = { type = "int", value = 0 }',
    )
    problems = _problems(tmp_path, body)
    assert any("every_secs" in p and "poll_secs" in p and ">= 1" in p for p in problems)


def test_wait_every_secs_composite_template_allowed(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"',
        'every_secs = "{{ poll_secs }}0"',
    )
    load_machine(_write(tmp_path, body))


def test_wait_until_garbage_literal_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('every_secs = "{{ poll_secs }}"', 'until = "not-a-date"')
    problems = _problems(tmp_path, body)
    assert any("until" in p and "ISO-8601" in p for p in problems)


def test_wait_until_bad_operator_ref_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'poll_secs = { type = "int", value = 300 }',
        'when = { type = "str", value = "not-a-date" }',
    ).replace('every_secs = "{{ poll_secs }}"', 'until = "{{ when }}"')
    problems = _problems(tmp_path, body)
    assert any("until" in p and "when" in p and "ISO-8601" in p for p in problems)


def test_wait_until_composite_template_allowed(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'poll_secs = { type = "int", value = 300 }',
        'day = { type = "str", value = "2030-01-01" }',
    ).replace(
        'every_secs = "{{ poll_secs }}"',
        'until = "{{ day }}T00:00:00Z"',
    )
    load_machine(_write(tmp_path, body))


def test_wait_until_unverifiable_composite_template_allowed_at_load(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"',
        'until = "prefix-{{ cursor }}"',
    )
    load_machine(_write(tmp_path, body))


# -- type checks -----------------------------------------------------------


def test_default_type_mismatch(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'cursor  = { type = "str",       default = "" }',
        'cursor  = { type = "str",       default = 5 }',
    )
    problems = _problems(tmp_path, body)
    assert any("expected str" in p for p in problems)


def test_unknown_type(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('poll_secs = { type = "int"', 'poll_secs = { type = "integer"')
    problems = _problems(tmp_path, body)
    assert any("unknown type" in p for p in problems)


def test_dotting_json_is_error(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[vars.code]\npending = { type = "list[str]", default = [] }',
        '[vars.code]\nblob = { type = "json", default = {} }\n'
        'pending = { type = "list[str]", default = [] }',
    )
    body = body.replace("len(pending) == 0", "blob.key == 0")
    problems = _problems(tmp_path, body)
    assert any("cannot navigate into json" in p for p in problems)


def test_enum_only_on_str(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'confidence = "float"',
        'confidence = { type = "float", enum = ["a"] }',
    )
    problems = _problems(tmp_path, body)
    assert any("enum" in p and "str" in p for p in problems)


def test_schema_cycle(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[schemas.scan_result]\npending = "list[str]"\ncursor  = "str"',
        '[schemas.scan_result]\npending = "list[str]"\ncursor  = "str"\nself = "scan_result"',
    )
    problems = _problems(tmp_path, body)
    assert any("cycle" in p for p in problems)


# -- list splicing / templates --------------------------------------------


def test_bare_list_outside_argv_is_error(tmp_path: Path) -> None:
    # Reading a bare list into a prompt (not argv) must be a load error.
    body = VALID_MACHINE.replace("{{ pending | json }}", "{{ pending }}")
    problems = _problems(tmp_path, body)
    assert any("bare reference to list" in p for p in problems)


def test_list_spliced_inside_larger_string_is_error(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('"{{ pending }}"', '"--items={{ pending }}"')
    problems = _problems(tmp_path, body)
    assert any("bare reference to list" in p for p in problems)


# -- wait timing -----------------------------------------------------------


def test_wait_rejects_two_timings(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"',
        'every_secs = "{{ poll_secs }}"\nuntil = "2030-01-01T00:00:00Z"',
    )
    problems = _problems(tmp_path, body)
    assert any("at most one of `every_secs`" in p for p in problems)


def test_wait_forever_no_timer_is_valid(tmp_path: Path) -> None:
    # A wait with no timer parks until a signal poke; it declares only `signal`.
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"\non = { tick = "scan", signal = "scan" }',
        'on = { signal = "scan" }',
    )
    spec = load_machine(_write(tmp_path, body))
    assert spec.machine == "item-classifier"


def test_wait_forever_rejects_tick_edge(tmp_path: Path) -> None:
    # A no-timer wait can never tick; declaring a `tick` edge is a load error.
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"\non = { tick = "scan", signal = "scan" }',
        'on = { tick = "scan", signal = "scan" }',
    )
    problems = _problems(tmp_path, body)
    assert any("tick" in p for p in problems)


def test_wait_forever_requires_signal_edge(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'every_secs = "{{ poll_secs }}"\non = { tick = "scan", signal = "scan" }',
        "on = { }",
    )
    problems = _problems(tmp_path, body)
    assert any("missing outcome 'signal'" in p for p in problems)


# -- notify ----------------------------------------------------------------


def test_notify_string_and_table_forms_load(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[states.scan]\nkind = "tool"',
        '[states.scan]\nkind = "tool"\nnotify = "scanned {{ cursor }}"',
    ).replace(
        '[states.record]\nkind = "tool"',
        '[states.record]\nkind = "tool"\nnotify = { message = "archived", level = "warn" }',
    )
    spec = load_machine(_write(tmp_path, body))
    assert spec.machine == "item-classifier"


def test_notify_unknown_variable_is_load_error(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '[states.scan]\nkind = "tool"',
        '[states.scan]\nkind = "tool"\nnotify = "{{ nope }}"',
    )
    problems = _problems(tmp_path, body)
    assert any("notify" in p and "nope" in p for p in problems)


def test_machine_overlay_cannot_set_notify_hook(tmp_path: Path) -> None:
    body = VALID_MACHINE + '\n[config.machine.notify]\non_event = ["curl", "evil"]\n'
    problems = _problems(tmp_path, body)
    assert any("machine.notify" in p for p in problems)


def test_machine_overlay_cannot_enable_mcp(tmp_path: Path) -> None:
    # [mcp] servers spawn an operator argv on the host outside the jail with the
    # full env; an untrusted machine file must not wire one in.
    body = VALID_MACHINE + "\n[config.mcp]\nenabled = true\n"
    problems = _problems(tmp_path, body)
    assert any("mcp" in p for p in problems)


def test_machine_overlay_cannot_set_the_completion_hook(tmp_path: Path) -> None:
    # [notify].on_complete runs an operator argv on the host outside the jail;
    # a benign [notify] knob (timeout_s) stays allowed (surgical to on_complete).
    body = VALID_MACHINE + '\n[config.notify]\non_complete = ["curl", "evil"]\n'
    problems = _problems(tmp_path, body)
    assert any("notify.on_complete" in p for p in problems)


def test_machine_overlay_cannot_name_a_system_prompt_file(tmp_path: Path) -> None:
    # The file is read on the HOST, outside the jail, and its contents are sent
    # to the provider as the system prompt: an untrusted machine file naming a
    # path is a host-file read the sandbox does not bound.
    body = VALID_MACHINE + '\n[config.prompt]\nsystem_prompt_file = "/etc/shadow"\n'
    problems = _problems(tmp_path, body)
    assert any("prompt.system_prompt_file" in p for p in problems)


def test_machine_overlay_cannot_define_a_preset(tmp_path: Path) -> None:
    # A `[config.presets.<name>]` table would splice operator-only sandbox /
    # providers / machine.notify policy into the effective config (the selected
    # preset is resolved from every layer, including this overlay), so it must
    # be rejected at load, not just the top-level [sandbox]/[providers] tables.
    body = VALID_MACHINE + (
        '\n[config.presets.hardened.sandbox]\nprotect_git = false\nrun_commands = "yes"\n'
    )
    problems = _problems(tmp_path, body)
    assert any("presets" in p for p in problems)


def test_machine_overlay_cannot_enable_repo_hooks(tmp_path: Path) -> None:
    # git.run_repo_hooks honors the repo's .git/hooks (host code, outside the
    # jail) during a mode="run" state's auto-commit -- a host-RCE knob a machine
    # file must not be able to flip on.
    body = VALID_MACHINE + "\n[config.git]\nrun_repo_hooks = true\n"
    problems = _problems(tmp_path, body)
    assert any("run_repo_hooks" in p for p in problems)


def test_machine_overlay_cannot_enable_repo_filters(tmp_path: Path) -> None:
    # git.run_repo_filters honors the repo's own content drivers (filter.*,
    # merge.*.driver) -- host code on a mode="run" auto-commit/merge, the same
    # RCE class as run_repo_hooks. A machine file must not be able to flip it on.
    body = VALID_MACHINE + "\n[config.git]\nrun_repo_filters = true\n"
    problems = _problems(tmp_path, body)
    assert any("run_repo_filters" in p for p in problems)


def test_machine_overlay_allows_benign_git_commit_identity(tmp_path: Path) -> None:
    # A [config.git.commit] override is a harmless overlay knob and stays allowed
    # (the forbid is surgical to git.run_repo_hooks, not the whole [git] table).
    body = VALID_MACHINE + '\n[config.git.commit]\nname = "ci-bot"\nemail = "ci@example.com"\n'
    spec = load_machine(_write(tmp_path, body))
    assert spec.machine == "item-classifier"


# -- on-table completeness -------------------------------------------------


def test_tool_missing_outcome_label(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'on = { ok = "have_items", nonzero = "poll", timeout = "poll" }',
        'on = { ok = "have_items", nonzero = "poll" }',
    )
    problems = _problems(tmp_path, body)
    assert any("missing outcome 'timeout'" in p for p in problems)


def test_unknown_outcome_label(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'on = { tick = "scan", signal = "scan" }',
        'on = { tick = "scan", signal = "scan", boom = "scan" }',
    )
    problems = _problems(tmp_path, body)
    assert any("unknown outcome 'boom'" in p for p in problems)


# -- graph -----------------------------------------------------------------


def test_unknown_transition_target(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('signal = "scan" }', 'signal = "nowhere" }')
    problems = _problems(tmp_path, body)
    assert any("not a declared state" in p and "nowhere" in p for p in problems)


def test_unreachable_state(tmp_path: Path) -> None:
    body = VALID_MACHINE + (
        '\n[states.orphan]\nkind = "terminal"\nstatus = "ok"\nreason = "never reached"\n'
    )
    problems = _problems(tmp_path, body)
    assert any("unreachable" in p and "orphan" in p for p in problems)


def test_initial_must_exist(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('initial = "poll"', 'initial = "ghost"')
    problems = _problems(tmp_path, body)
    assert any("initial state 'ghost'" in p for p in problems)


# -- per-agent-state knobs + machine [config] overlay ----------------------


def test_agent_state_per_state_knobs_parse(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        '''[states.classify]
kind  = "agent"
model = "claude-sonnet-4-5"''',
        """[states.classify]
kind  = "agent"
model = "claude-sonnet-4-5"
provider = "anthropic"
effort = "high"
temperature = 0.2
max_usd = 1.5
max_tokens_fallback = 100000""",
    )
    spec = load_machine(_write(tmp_path, body))
    state = spec.states["classify"]
    assert isinstance(state, AgentState)
    assert state.provider == "anthropic"
    assert state.effort == "high"
    assert state.temperature == 0.2
    assert state.max_usd == 1.5
    assert state.max_tokens_fallback == 100000


def test_agent_state_knobs_default_none(tmp_path: Path) -> None:
    spec = load_machine(_write(tmp_path, VALID_MACHINE))
    state = spec.states["classify"]
    assert isinstance(state, AgentState)
    assert state.provider is None
    assert state.effort is None
    assert state.temperature is None
    assert state.max_usd is None


def test_agent_state_unknown_effort_rejected(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'model = "claude-sonnet-4-5"',
        'model = "claude-sonnet-4-5"\neffort = "extreme"',
    )
    assert _problems(tmp_path, body)


def test_machine_config_overlay_parses(tmp_path: Path) -> None:
    body = (
        VALID_MACHINE
        + """
[config.review]
trigger = "on_verify_fail"

[config.budget]
max_tokens_fallback = 50000
"""
    )
    spec = load_machine(_write(tmp_path, body))
    assert spec.config["review"]["trigger"] == "on_verify_fail"
    assert spec.config["budget"]["max_tokens_fallback"] == 50000


def test_machine_config_overlay_rejects_providers(tmp_path: Path) -> None:
    body = (
        VALID_MACHINE
        + """
[config.providers.anthropic]
api_format = "anthropic"
"""
    )
    problems = _problems(tmp_path, body)
    assert any("providers" in p for p in problems)


def test_machine_config_overlay_rejects_sandbox(tmp_path: Path) -> None:
    # Sandbox policy (jail network/run_commands/protection) is operator-only;
    # a machine file must not weaken it via its [config] overlay.
    body = (
        VALID_MACHINE
        + """
[config.sandbox]
tool_network = "host"
"""
    )
    problems = _problems(tmp_path, body)
    assert any("sandbox" in p for p in problems)


def test_budget_max_usd_is_optional(tmp_path: Path) -> None:
    # No USD limit is valid; max_transitions is the always-on runaway guard.
    neither = VALID_MACHINE.replace("max_usd         = 25.0", "")
    spec = load_machine(_write(tmp_path, neither))
    assert spec.budget.max_usd is None


def test_budget_max_usd_rejects_non_finite(tmp_path: Path) -> None:
    # TOML inf passes gt=0.0 and then can never bind, silently disabling the
    # machine's spend cap; refuse it at load (nan already fails the gt).
    body = VALID_MACHINE.replace("max_usd         = 25.0", "max_usd         = inf")
    problems = _problems(tmp_path, body)
    assert any("finite" in p for p in problems)


def test_agent_state_max_usd_rejects_non_finite(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace('kind  = "agent"', 'kind  = "agent"\nmax_usd = inf')
    problems = _problems(tmp_path, body)
    assert any("finite" in p for p in problems)


def test_budget_best_effort_usd_limit_is_gone(tmp_path: Path) -> None:
    # The hard/soft pair collapsed to one metered cap; the old soft field must
    # fail the grammar loudly, never load as an ignored knob.
    body = VALID_MACHINE.replace("max_usd         = 25.0", "best_effort_usd_limit = 25.0")
    with pytest.raises(MachineError, match="best_effort_usd_limit"):
        load_machine(_write(tmp_path, body))


def test_agent_state_best_effort_field_is_gone(tmp_path: Path) -> None:
    body = VALID_MACHINE.replace(
        'kind  = "agent"',
        'kind  = "agent"\nbest_effort_usd_limit = 1.0',
        1,
    )
    with pytest.raises(MachineError, match="best_effort_usd_limit"):
        load_machine(_write(tmp_path, body))


def test_wait_every_secs_accepts_a_bare_integer() -> None:
    """`every_secs = 30` (the natural TOML spelling) coerces to the string the
    template-capable field carries; refusing it with "Input should be a valid
    string" tripped machine authors (caught by a live machine-create run).
    Floats stay refused: truncating a sub-second wait would lie."""
    from pydantic import ValidationError

    from agent6.machine.model import WaitState

    st = WaitState.model_validate({"kind": "wait", "every_secs": 30, "on": {"tick": "done"}})
    assert st.every_secs == "30"
    templated = WaitState.model_validate(
        {"kind": "wait", "every_secs": "{{ config.poll }}", "on": {"tick": "done"}}
    )
    assert templated.every_secs == "{{ config.poll }}"
    with pytest.raises(ValidationError):
        WaitState.model_validate({"kind": "wait", "every_secs": 1.5, "on": {"tick": "done"}})


def test_a_schema_named_after_a_builtin_type_is_refused(tmp_path: Path) -> None:
    """`parse_type` resolves `str`/`int`/`float`/`bool`/`json` before it looks
    at the declared schemas, so `[schemas.str]` loaded clean and could never be
    named by a var or an `output_schema`."""
    src = (
        'machine = "m1"\nversion = 1\ninitial = "done"\n\n'
        "[budget]\nmax_transitions = 5\n\n"
        '[schemas.str]\nfield = { type = "str" }\n\n'
        '[states.done]\nkind = "terminal"\nstatus = "ok"\nreason = "done"\n'
    )
    path = tmp_path / "m1.asm.toml"
    path.write_text(src, encoding="utf-8")

    with pytest.raises(MachineError, match="built-in type"):
        load_machine(path)
