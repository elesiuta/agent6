# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Load + semantic validation for `.asm.toml` state machines.

Parsing is pydantic (the shapes in `model`); this module adds the cross-
cutting load-time rules the spec requires (global name uniqueness, the
ownership wall, reference/field type checks, total branches, reachability)
plus `load_machine` and `validate_record_payload`. Every violation is a
load-time error aggregated into MachineError.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from agent6.machine.model import (
    AGENT_LABELS,
    BUILTIN_TYPE_NAMES,
    IDENT_RE,
    RESERVED_NAMES,
    TOOL_LABELS,
    WAIT_LABELS,
    AgentState,
    BranchState,
    Capture,
    FieldSpec,
    JsonT,
    ListT,
    MachineError,
    MachineSpec,
    RecordT,
    ScalarT,
    StateSpec,
    ToolState,
    TypeParseError,
    TypeRef,
    WaitState,
    edges,
    parse_type,
    reachable_states,
    type_decl,
    type_str,
)
from agent6.machine.predicate import (
    PredicateError,
    Reference,
    parse_predicate,
)
from agent6.machine.template import (
    Interp,
    Template,
    TemplateError,
    parse_template,
)


def load_machine(path: Path) -> MachineSpec:
    """Load, parse, and fully validate the `.asm.toml` file at *path*.

    Raises :class:`MachineError` aggregating every diagnostic. Never
    returns a partially-valid machine.
    """
    if not path.is_file():
        raise MachineError([f"machine file not found: {path}"])
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise MachineError([f"not valid TOML ({path}): {exc}"]) from exc
    except UnicodeDecodeError as exc:
        raise MachineError([f"not valid UTF-8 ({path}): {exc}"]) from exc
    except OSError as exc:
        raise MachineError([f"cannot be read ({path}): {exc}"]) from exc
    precheck = _precheck(raw)
    if precheck:
        raise MachineError(precheck)
    try:
        spec = MachineSpec.model_validate(raw)
    except ValidationError as exc:
        raise MachineError(_format_validation_error(exc)) from exc
    problems = validate_semantics(spec)
    if problems:
        raise MachineError(problems)
    return spec


def _precheck(raw: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    vars_section = raw.get("vars")
    if isinstance(vars_section, dict):
        for key in vars_section:
            if key not in ("operator", "code", "agent"):
                problems.append(
                    f"`vars.{key}` has no owner subtable; put it in"
                    " `[vars.operator]`, `[vars.code]`, or `[vars.agent]`"
                )
    return problems


def _format_validation_error(err: ValidationError) -> list[str]:
    problems: list[str] = []
    for issue in err.errors():
        loc = ".".join(str(part) for part in issue["loc"]) or "<root>"
        problems.append(f"{loc}: {issue['msg']} (type={issue['type']})")
    return problems


# --------------------------------------------------------------------------
# Semantic validation (§4.5)
# --------------------------------------------------------------------------


def validate_semantics(spec: MachineSpec) -> list[str]:
    """Run every cross-cutting rule the pydantic shape cannot express."""
    problems: list[str] = []
    schema_names = frozenset(spec.schemas)

    for sname in spec.schemas:
        if not IDENT_RE.match(sname):
            problems.append(f"schema name {sname!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")
        elif sname in BUILTIN_TYPE_NAMES:
            # `parse_type` resolves the built-ins before the schema names, so a
            # schema called `str` or `json` loads clean and can never be named
            # by a var or an `output_schema`.
            problems.append(
                f"schema name {sname!r} is a built-in type, so nothing could reference it"
            )

    schemas, schema_problems = _resolve_schemas(spec, schema_names)
    problems.extend(schema_problems)

    var_types, var_owner, var_values, var_problems = _resolve_vars(spec, schema_names, schemas)
    problems.extend(var_problems)

    if spec.initial not in spec.states:
        problems.append(f"initial state {spec.initial!r} is not a declared state")

    for name, state in spec.states.items():
        if not IDENT_RE.match(name):
            problems.append(f"state name {name!r} is not a valid identifier (^[a-z][a-z0-9_]*$)")
        problems.extend(
            _validate_state(name, state, var_types, var_owner, var_values, schemas, schema_names)
        )

    problems.extend(_validate_graph(spec))
    return problems


def _resolve_schemas(
    spec: MachineSpec, schema_names: frozenset[str]
) -> tuple[dict[str, dict[str, TypeRef]], list[str]]:
    problems: list[str] = []
    resolved: dict[str, dict[str, TypeRef]] = {}
    for sname, fields in spec.schemas.items():
        resolved_fields: dict[str, TypeRef] = {}
        for fname, field in fields.items():
            if not IDENT_RE.match(fname):
                problems.append(f"schema {sname!r}: field name {fname!r} is not a valid identifier")
            try:
                ftype = parse_type(field.type, schema_names)
            except TypeParseError as exc:
                problems.append(f"schema {sname!r}.{fname}: {exc}")
                continue
            if field.enum is not None and ftype != ScalarT("str"):
                problems.append(f"schema {sname!r}.{fname}: `enum` is only valid on `str` fields")
            resolved_fields[fname] = ftype
        resolved[sname] = resolved_fields
    problems.extend(_detect_schema_cycles(resolved))
    return resolved, problems


def _detect_schema_cycles(resolved: dict[str, dict[str, TypeRef]]) -> list[str]:
    problems: list[str] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in done or name not in resolved:
            return
        if name in visiting:
            cycle = " -> ".join((*trail, name))
            problems.append(f"record schema cycle: {cycle}")
            return
        visiting.add(name)
        for ftype in resolved[name].values():
            if isinstance(ftype, RecordT):
                visit(ftype.name, (*trail, name))
        visiting.discard(name)
        done.add(name)

    for name in resolved:
        visit(name, ())
    return problems


def _resolve_vars(
    spec: MachineSpec,
    schema_names: frozenset[str],
    schemas: dict[str, dict[str, TypeRef]],
) -> tuple[dict[str, TypeRef], dict[str, str], dict[str, Any], list[str]]:
    problems: list[str] = []
    var_types: dict[str, TypeRef] = {}
    var_owner: dict[str, str] = {}
    var_values: dict[str, Any] = {}

    declared: dict[str, str] = {}
    owners: tuple[tuple[str, dict[str, Any]], ...] = (
        ("operator", dict(spec.vars.operator)),
        ("code", dict(spec.vars.code)),
        ("agent", dict(spec.vars.agent)),
    )
    for owner, table in owners:
        for vname, varspec in table.items():
            if not IDENT_RE.match(vname):
                problems.append(
                    f"variable name {vname!r} in `[vars.{owner}]` is not a valid identifier"
                    " (^[a-z][a-z0-9_]*$)"
                )
            if vname in RESERVED_NAMES:
                problems.append(f"variable name {vname!r} is reserved and may not be used")
            if vname in declared:
                problems.append(
                    f"variable {vname!r} declared in both `[vars.{declared[vname]}]` and"
                    f" `[vars.{owner}]`; the three owner subtables share one read namespace"
                )
                continue
            declared[vname] = owner
            var_owner[vname] = owner
            try:
                vtype = parse_type(varspec.type, schema_names)
            except TypeParseError as exc:
                problems.append(f"variable {vname!r} in `[vars.{owner}]`: {exc}")
                continue
            var_types[vname] = vtype
            value = varspec.value if owner == "operator" else varspec.default
            var_values[vname] = value
            problems.extend(_check_value(value, vtype, schemas, f"variable {vname!r}"))
    return var_types, var_owner, var_values, problems


def fixture_problems(spec: MachineSpec, fixture: dict[str, Any]) -> list[str]:
    """Problems in a `--blackboard` fixture: every key must name a declared
    var and every value must satisfy its type -- the same checks the declared
    defaults get. Unchecked, a typo'd key was silently ignored and a string
    `"false"` replaced a bool and routed branches as truthy."""
    schema_names = frozenset(spec.schemas)
    schemas, _ = _resolve_schemas(spec, schema_names)
    var_types, _owner, _values, _ = _resolve_vars(spec, schema_names, schemas)
    problems: list[str] = []
    for name, value in fixture.items():
        if name not in var_types:
            problems.append(f"blackboard fixture: {name!r} is not a declared variable")
            continue
        problems.extend(_check_value(value, var_types[name], schemas, f"fixture {name!r}"))
    return problems


def _check_value(
    value: Any, t: TypeRef, schemas: dict[str, dict[str, TypeRef]], label: str
) -> list[str]:
    if isinstance(t, ScalarT):
        return _check_scalar(value, t.name, label)
    if isinstance(t, ListT):
        if not isinstance(value, list):
            return [f"{label}: expected list, got {_py_type(value)}"]
        problems: list[str] = []
        for index, element in enumerate(value):
            problems.extend(_check_scalar(element, t.elem, f"{label}[{index}]"))
        return problems
    if isinstance(t, JsonT):
        return _check_json(value, label)
    # RecordT: a default/value is a placeholder (the example uses `{}` for a
    # required-field record), so the check does not require presence, but any field
    # that *is* present must be known and well-typed.
    if not isinstance(value, dict):
        return [f"{label}: expected object for record {t.name!r}, got {_py_type(value)}"]
    problems = []
    fields = schemas.get(t.name, {})
    for key, sub in value.items():
        if not isinstance(key, str) or key not in fields:
            problems.append(f"{label}: unknown field {key!r} for record {t.name!r}")
            continue
        problems.extend(_check_value(sub, fields[key], schemas, f"{label}.{key}"))
    return problems


def _check_scalar(value: Any, name: str, label: str) -> list[str]:
    if name == "bool":
        ok = isinstance(value, bool)
    elif name == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif name == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:  # str
        ok = isinstance(value, str)
    if not ok:
        return [f"{label}: expected {name}, got {_py_type(value)}"]
    return []


def _check_json(value: Any, label: str) -> list[str]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return []
    if isinstance(value, list):
        problems: list[str] = []
        for index, element in enumerate(value):
            problems.extend(_check_json(element, f"{label}[{index}]"))
        return problems
    if isinstance(value, dict):
        problems = []
        for key, sub in value.items():
            if not isinstance(key, str):
                problems.append(f"{label}: json object keys must be strings, got {_py_type(key)}")
            problems.extend(_check_json(sub, f"{label}.{key}"))
        return problems
    return [f"{label}: value is not JSON-serializable ({_py_type(value)})"]


def _py_type(value: Any) -> str:
    return type(value).__name__


# --------------------------------------------------------------------------
# Runtime payload validation (the agent `finish_session` trust boundary)
# --------------------------------------------------------------------------


def validate_record_payload(
    schemas: dict[str, dict[str, FieldSpec]], schema_name: str, payload: Any, *, where: str
) -> list[str]:
    """Strictly validate a captured *payload* against a record schema.

    The one runtime capture gate for an agent `finish_session` payload (engine
    side AND the leg's in-run contract check, which receives the same schemas
    over the AgentRequest wire) and a `tool` state's parsed stdout (*where*
    names which, for the error text). Stricter than the load-time placeholder
    check on variable defaults: every non-optional field must be present,
    `enum` constraints are enforced, and nested records recurse. Presumes the
    schemas already passed `validate_semantics`, so the graph is well-formed.
    Returns an empty list when the payload conforms, or a list of
    human-readable problems otherwise.
    """
    return _check_record_strict(payload, schema_name, schemas, frozenset(schemas), where)


def _check_record_strict(
    value: Any,
    schema_name: str,
    raw_schemas: dict[str, dict[str, FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    fields = raw_schemas.get(schema_name)
    if fields is None:
        return [f"{label}: unknown schema {schema_name!r}"]
    if not isinstance(value, dict):
        return [f"{label}: expected object for record {schema_name!r}, got {_py_type(value)}"]
    problems: list[str] = []
    for fname, field in fields.items():
        if fname not in value:
            if not field.optional:
                problems.append(f"{label}: missing required field {fname!r}")
            continue
        problems.extend(
            _check_field_value(value[fname], field, raw_schemas, schema_names, f"{label}.{fname}")
        )
    for key in value:
        if key not in fields:
            problems.append(f"{label}: unknown field {key!r} for record {schema_name!r}")
    return problems


def _check_field_value(
    value: Any,
    field: FieldSpec,
    raw_schemas: dict[str, dict[str, FieldSpec]],
    schema_names: frozenset[str],
    label: str,
) -> list[str]:
    try:
        ftype = parse_type(field.type, schema_names)
    except TypeParseError as exc:  # pragma: no cover - spec already validated
        return [f"{label}: {exc}"]
    if isinstance(ftype, RecordT):
        return _check_record_strict(value, ftype.name, raw_schemas, schema_names, label)
    problems = _check_value(value, ftype, {}, label)
    if not problems and field.enum is not None and value not in field.enum:
        problems.append(f"{label}: {value!r} is not one of enum {list(field.enum)}")
    return problems


# --------------------------------------------------------------------------
# Reference / template resolution
# --------------------------------------------------------------------------


def _resolve_ref_type(
    ref: Reference,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
) -> tuple[TypeRef | None, str | None]:
    if ref.root == "result":
        if result_type is None:
            return None, f"`result` is not navigable here ({ref.dotted!r})"
        current: TypeRef = result_type
    else:
        looked_up = var_types.get(ref.root)
        if looked_up is None:
            return None, f"unknown variable {ref.root!r}"
        current = looked_up
    for key in ref.path:
        if not isinstance(current, RecordT):
            return None, f"cannot navigate into {type_str(current)} at {ref.dotted!r}"
        fields = schemas.get(current.name, {})
        if key not in fields:
            return None, f"record {current.name!r} has no field {key!r} (in {ref.dotted!r})"
        current = fields[key]
    return current, None


def _validate_template(
    text: str,
    *,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
    allow_splice: bool,
    where: str,
) -> list[str]:
    try:
        template = parse_template(text)
    except TemplateError as exc:
        return [f"{where}: {exc}"]
    problems: list[str] = []
    for part in template.parts:
        if not isinstance(part, Interp):
            continue
        ref_type, error = _resolve_ref_type(part.ref, var_types, schemas, result_type)
        if error is not None:
            problems.append(f"{where}: {error}")
            continue
        assert ref_type is not None
        problems.extend(
            _check_interp_filter(part, ref_type, template, allow_splice=allow_splice, where=where)
        )
    return problems


def _check_interp_filter(
    part: Interp,
    ref_type: TypeRef,
    template: Template,
    *,
    allow_splice: bool,
    where: str,
) -> list[str]:
    if part.filt == "json":
        return []
    if part.filt == "len":
        if isinstance(ref_type, ScalarT) and ref_type.name != "str":
            return [
                f"{where}: `| len` does not apply to {type_str(ref_type)} ({part.ref.dotted!r})"
            ]
        return []
    # Bare reference (no filter): must be a scalar, unless it is a lone
    # list reference spliced into argv (§4.4).
    if isinstance(ref_type, ScalarT):
        return []
    if allow_splice and isinstance(ref_type, ListT) and template.is_lone_ref:
        return []
    return [
        f"{where}: bare reference to {type_str(ref_type)} ({part.ref.dotted!r});"
        " apply `| json` or, for a list in argv, splice it as a standalone element"
    ]


# --------------------------------------------------------------------------
# State validation
# --------------------------------------------------------------------------


def _validate_state(
    name: str,
    state: StateSpec,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    var_values: dict[str, Any],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    # `notify` is on every kind (§4.3): validate its message template up front.
    problems: list[str] = []
    if state.notify is not None:
        problems.extend(
            _validate_template(
                state.notify.message,
                var_types=var_types,
                schemas=schemas,
                result_type=None,
                allow_splice=False,
                where=f"state {name!r} notify",
            )
        )
    if isinstance(state, AgentState):
        problems.extend(_validate_agent(name, state, var_types, var_owner, schemas, schema_names))
    elif isinstance(state, ToolState):
        problems.extend(_validate_tool(name, state, var_types, var_owner, schemas, schema_names))
    elif isinstance(state, WaitState):
        problems.extend(_validate_wait(name, state, var_types, var_owner, var_values, schemas))
    elif isinstance(state, BranchState):
        problems.extend(_validate_branch(name, state, var_types, schemas))
    return problems  # TerminalState: shape is fully checked by pydantic


def _validate_on(name: str, on: dict[str, str], expected: frozenset[str]) -> list[str]:
    got = frozenset(on)
    problems: list[str] = []
    for missing in sorted(expected - got):
        problems.append(f"state {name!r}: `on` is missing outcome {missing!r}")
    for extra in sorted(got - expected):
        problems.append(
            f"state {name!r}: `on` has unknown outcome {extra!r} (allowed: {sorted(expected)})"
        )
    return problems


def _validate_agent(
    name: str,
    state: AgentState,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    problems = _validate_on(name, state.on, AGENT_LABELS)
    if state.output_schema not in schema_names:
        problems.append(
            f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
        )
        result_type: TypeRef | None = None
    else:
        result_type = RecordT(state.output_schema)
    problems.extend(
        _validate_template(
            state.prompt,
            var_types=var_types,
            schemas=schemas,
            result_type=None,
            allow_splice=False,
            where=f"state {name!r} prompt",
        )
    )
    if state.capture.stdout_json is not None:
        problems.append(f"state {name!r}: an `agent` capture uses `finish_json`, not `stdout_json`")
    problems.extend(
        _validate_capture(
            name,
            state.capture,
            owner="agent",
            var_types=var_types,
            var_owner=var_owner,
            schemas=schemas,
            result_type=result_type,
            whole_type=result_type,
        )
    )
    return problems


def _validate_tool(
    name: str,
    state: ToolState,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    schema_names: frozenset[str],
) -> list[str]:
    problems = _validate_on(name, state.on, TOOL_LABELS)
    result_type: TypeRef | None = None
    if state.output_schema is not None:
        if state.output_schema not in schema_names:
            problems.append(
                f"state {name!r}: output_schema {state.output_schema!r} is not a declared schema"
            )
        else:
            result_type = RecordT(state.output_schema)
    for index, element in enumerate(state.command):
        problems.extend(
            _validate_template(
                element,
                var_types=var_types,
                schemas=schemas,
                result_type=None,
                allow_splice=True,
                where=f"state {name!r} command[{index}]",
            )
        )
    if state.capture is not None:
        if state.capture.finish_json is not None:
            problems.append(
                f"state {name!r}: a `tool` capture uses `stdout_json`, not `finish_json`"
            )
        if state.capture.stdout_json is not None and state.output_schema is not None:
            problems.append(
                f"state {name!r}: `stdout_json` whole-capture is opaque; drop `output_schema`"
                " or use `set` field-capture"
            )
        problems.extend(
            _validate_capture(
                name,
                state.capture,
                owner="code",
                var_types=var_types,
                var_owner=var_owner,
                schemas=schemas,
                result_type=result_type,
                whole_type=JsonT(),
            )
        )
    return problems


def _validate_capture(
    name: str,
    capture: Capture,
    *,
    owner: str,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
    whole_type: TypeRef | None,
) -> list[str]:
    problems: list[str] = []
    whole_target = capture.stdout_json if owner == "code" else capture.finish_json
    if whole_target is not None:
        problems.extend(_check_capture_target(name, whole_target, owner, var_owner))
        target_type = var_types.get(whole_target)
        if target_type is not None and whole_type is not None and target_type != whole_type:
            problems.append(
                f"state {name!r}: capture target {whole_target!r} has type"
                f" {type_str(target_type)} but the captured value is {type_str(whole_type)};"
                f' declare it as type = "{type_decl(whole_type)}"'
            )
    if capture.set is not None:
        for target, template in capture.set.items():
            problems.extend(_check_capture_target(name, target, owner, var_owner))
            problems.extend(
                _validate_set_assignment(
                    name,
                    target,
                    template,
                    var_types=var_types,
                    schemas=schemas,
                    result_type=result_type,
                )
            )
    return problems


def _check_capture_target(
    name: str, target: str, owner: str, var_owner: dict[str, str]
) -> list[str]:
    actual = var_owner.get(target)
    if actual is None:
        return [
            f"state {name!r}: capture target {target!r} is not a declared variable;"
            f" declare it in [vars.{owner}]"
        ]
    if actual != owner:
        return [
            f"state {name!r}: a `{owner}` state may only write `[vars.{owner}]` variables,"
            f" but {target!r} is owned by `[vars.{actual}]`"
        ]
    return []


def _validate_set_assignment(
    name: str,
    target: str,
    template: str,
    *,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
    result_type: TypeRef | None,
) -> list[str]:
    where = f"state {name!r} capture.set.{target}"
    try:
        parsed = parse_template(template)
    except TemplateError as exc:
        return [f"{where}: {exc}"]
    target_type = var_types.get(target)
    # A lone, filter-less interpolation captures the referenced *value* with
    # its native type (the only way a non-string value reaches the
    # blackboard); its type must match the target variable.
    if parsed.is_lone_ref:
        interp = parsed.parts[0]
        assert isinstance(interp, Interp)
        source_type, error = _resolve_ref_type(interp.ref, var_types, schemas, result_type)
        if error is not None:
            return [f"{where}: {error}"]
        if target_type is not None and source_type is not None and source_type != target_type:
            return [
                f"{where}: assigns {type_str(source_type)} to {target!r} of type"
                f" {type_str(target_type)};"
                f' declare it as type = "{type_decl(source_type)}"'
            ]
        return []
    # Otherwise the assignment renders to a string, so the target must be str.
    problems = _validate_template(
        template,
        var_types=var_types,
        schemas=schemas,
        result_type=result_type,
        allow_splice=False,
        where=where,
    )
    if target_type is not None and target_type != ScalarT("str"):
        problems.append(
            f"{where}: a rendered template yields a string but {target!r} has type"
            f' {type_str(target_type)}; declare it as type = "str"'
            " (only a lone {{ var }} keeps a value's type)"
        )
    return problems


def _validate_wait(
    name: str,
    state: WaitState,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    var_values: dict[str, Any],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    timings = [
        timing
        for timing, value in (
            ("every_secs", state.every_secs),
            ("until", state.until),
        )
        if value is not None
    ]
    if len(timings) > 1:
        problems = [
            f"state {name!r}: a `wait` may declare at most one of `every_secs` or"
            f" `until` (found: {timings})"
        ]
    elif not timings:
        # A wait with no timer parks until an operator `signal` poke (§4.3). It
        # can never `tick`, so it declares only `signal`; declaring `tick` is a
        # load error (an unreachable edge).
        problems = _validate_on(name, state.on, WAIT_LABELS - frozenset({"tick"}))
    else:
        problems = _validate_on(name, state.on, WAIT_LABELS)
    for timing, value in (
        ("every_secs", state.every_secs),
        ("until", state.until),
    ):
        if value is None:
            continue
        problems.extend(
            _validate_template(
                value,
                var_types=var_types,
                schemas=schemas,
                result_type=None,
                allow_splice=False,
                where=f"state {name!r} {timing}",
            )
        )
    # Value-validate the timing at load so `check`/`test` catch a float/garbage
    # `every_secs`, a busy-looping `every_secs <= 0`, or a non-ISO `until`, rather
    # than letting them surface only when the wait is first reached at run.
    if state.every_secs is not None:
        problems.extend(
            _timing_problems(
                name,
                "every_secs",
                state.every_secs,
                var_types,
                var_owner,
                var_values,
                schemas,
                kind="int",
            )
        )
    if state.until is not None:
        problems.extend(
            _timing_problems(
                name, "until", state.until, var_types, var_owner, var_values, schemas, kind="iso"
            )
        )
    return problems


def _timing_literal(text: str) -> str | None:
    """The constant value of a timing template, or None if it interpolates a
    variable (validated against the variable's type instead). Returns None on a
    malformed template, which `_validate_template` already reported."""
    try:
        template = parse_template(text)
    except TemplateError:
        return None
    if any(isinstance(part, Interp) for part in template.parts):
        return None
    return "".join(part for part in template.parts if isinstance(part, str))


def _timing_ref(text: str) -> Reference | None:
    """The lone reference in a timing template, else None.

    Malformed templates and composite templates return None. `_validate_template`
    already reports parse errors, and composite templates are data-dependent: the
    engine validates the rendered value at runtime.
    """
    try:
        template = parse_template(text)
    except TemplateError:
        return None
    if not template.is_lone_ref:
        return None
    interp = template.parts[0]
    assert isinstance(interp, Interp)
    return interp.ref


def _static_ref_value(ref: Reference, var_values: dict[str, Any]) -> Any:
    current: Any = var_values.get(ref.root)
    for key in ref.path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _timing_literal_problems(
    name: str, key: str, literal: str, kind: Literal["int", "iso"]
) -> list[str]:
    """Range/format-check a constant timing literal: `every_secs` (kind="int",
    a positive integer) or `until` (kind="iso", an ISO-8601 instant)."""
    if kind == "int":
        try:
            seconds = int(literal)
        except ValueError:
            return [
                f"state {name!r}: `{key}` must be an integer literal or an int variable"
                f" reference; got {literal!r}"
            ]
        if seconds < 1:
            return [
                f"state {name!r}: `{key}` must be >= 1; got {seconds}"
                " (a zero or negative interval would busy-loop)"
            ]
        return []
    try:
        datetime.fromisoformat(literal)
    except ValueError:
        return [
            f"state {name!r}: `{key}` must be an ISO-8601 instant or a str variable"
            f" reference; got {literal!r}"
        ]
    return []


def _timing_problems(
    name: str,
    key: str,
    text: str,
    var_types: dict[str, TypeRef],
    var_owner: dict[str, str],
    var_values: dict[str, Any],
    schemas: dict[str, dict[str, TypeRef]],
    *,
    kind: Literal["int", "iso"],
) -> list[str]:
    """Value-validate a wait timing template: `every_secs` (kind="int") or `until`
    (kind="iso"). A literal is range/format-checked; a lone variable reference is
    type-checked, and an operator-owned constant is checked statically. The two
    kinds share this reference-resolution skeleton and differ only in the parser,
    the required scalar type, and the constant check."""
    problems: list[str] = []
    literal = _timing_literal(text)
    if literal is not None:
        return _timing_literal_problems(name, key, literal, kind)

    ref = _timing_ref(text)
    if ref is None:
        return problems
    ref_type, error = _resolve_ref_type(ref, var_types, schemas, None)
    if error is None:
        assert ref_type is not None
        expected = ScalarT("int") if kind == "int" else ScalarT("str")
        if ref_type != expected:
            noun = "an int" if kind == "int" else "a str"
            problems.append(
                f"state {name!r}: `{key}` must reference {noun} variable, not {type_str(ref_type)}"
            )
        elif var_owner.get(ref.root) == "operator":
            value = _static_ref_value(ref, var_values)
            if kind == "int":
                if isinstance(value, int) and not isinstance(value, bool) and value < 1:
                    problems.append(
                        f"state {name!r}: `{key}` must be >= 1; {ref.dotted!r} is {value}"
                        " (a zero or negative interval would busy-loop)"
                    )
            elif isinstance(value, str):  # kind == "iso"
                try:
                    datetime.fromisoformat(value)
                except ValueError:
                    problems.append(
                        f"state {name!r}: `{key}` must be an ISO-8601 instant;"
                        f" {ref.dotted!r} is {value!r}"
                    )
    return problems


def _validate_branch(
    name: str,
    state: BranchState,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    problems: list[str] = []
    last_index = len(state.when) - 1
    for index, clause in enumerate(state.when):
        if clause.else_ is not None and index != last_index:
            problems.append(f"state {name!r}: an `else` clause must be the final `when` clause")
        if clause.if_ is not None:
            problems.extend(_validate_predicate(name, clause.if_, var_types, schemas))
    if state.when[last_index].else_ is None:
        problems.append(
            f"state {name!r}: branch is not total (no final `else`);"
            " add `{ else = true, goto = ... }`"
        )
    return problems


def _validate_predicate(
    name: str,
    source: str,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    try:
        predicate = parse_predicate(source)
    except PredicateError as exc:
        return [f"state {name!r}: predicate {source!r}: {exc}"]
    problems: list[str] = []
    for ref in predicate.references:
        _, error = _resolve_ref_type(ref, var_types, schemas, None)
        if error is not None:
            # The file is TOML, so an author naturally writes `flag == true`; but
            # predicates are Python-parsed, so `true` reads as an undeclared name.
            # Point at the Python literal rather than leaving a bare "unknown var".
            if not ref.path and ref.root in {"true", "false", "null", "none"}:
                error += (
                    " (predicates use Python literals True/False/None, not TOML"
                    " true/false/null; for a bool var write the bare name, e.g. `flag`)"
                )
            problems.append(f"state {name!r}: predicate {source!r}: {error}")
    # Type-check `len()` arguments at load (mirrors the template `| len` filter):
    # `len(n)` on an int/float/bool is a guaranteed runtime PredicateError, and
    # the spec promises type mismatches are load-time errors, not run surprises.
    problems.extend(_predicate_len_problems(name, source, predicate.tree.body, var_types, schemas))
    return problems


def _ast_reference(node: ast.expr) -> Reference | None:
    """Reconstruct a blackboard :class:`Reference` from an already-allow-listed
    predicate AST node (a `Name` or `Attribute` chain), or None for anything
    else. The structure was validated by `parse_predicate`."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return Reference(root=parts[0], path=tuple(parts[1:]))


def _predicate_len_problems(
    name: str,
    source: str,
    body: ast.expr,
    var_types: dict[str, TypeRef],
    schemas: dict[str, dict[str, TypeRef]],
) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(body):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
        ):
            continue
        arg = node.args[0]
        ref = _ast_reference(arg)
        if ref is not None:
            ftype, error = _resolve_ref_type(ref, var_types, schemas, None)
            if error is None and isinstance(ftype, ScalarT) and ftype.name != "str":
                problems.append(
                    f"state {name!r}: predicate {source!r}: `len()` does not apply to"
                    f" {type_str(ftype)} ({ref.dotted!r})"
                )
        elif isinstance(arg, ast.Constant) and not isinstance(arg.value, str):
            problems.append(
                f"state {name!r}: predicate {source!r}: `len()` does not apply to literal"
                f" {arg.value!r}"
            )
    return problems


def _validate_graph(spec: MachineSpec) -> list[str]:
    problems: list[str] = []
    for edge in edges(spec):
        if edge.dst not in spec.states:
            problems.append(
                f"state {edge.src!r}: transition target {edge.dst!r} is not a declared state"
            )
    reachable = reachable_states(spec)
    for name in spec.states:
        if name not in reachable:
            problems.append(f"state {name!r} is unreachable from initial state {spec.initial!r}")
    return problems
