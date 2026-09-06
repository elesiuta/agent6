# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Text-embedded tool-call recovery for the OpenAI-compatible provider.

Fallback parsing for models whose server does not populate the native
`tool_calls` array and instead leaks the call into the assistant
`content` text (Qwen/Hermes tags, Qwen-Coder XML, Gemma `tool_code`
fences, bare or fenced JSON). The rationale and the guards live on the
comment block below; `providers/_openai_parse.py`'s `parse_response` is the
only production caller.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

# Some OpenAI-compatible servers (notably certain Ollama / llama.cpp
# chat templates for Qwen, Hermes, and other small local models, and some
# OpenRouter upstream backends) do NOT parse the model's tool call into the
# native `tool_calls` array. Instead the call leaks into the assistant
# `content` as plain text, in one of several shapes:
#   - a bare JSON object `{"name": ..., "arguments": {...}}`,
#   - the same wrapped in a ```json fence,
#   - Hermes/Qwen `<tool_call>{json}</tool_call>` tags, or
#   - the Qwen-Coder XML form ``<function=NAME><parameter=KEY>VALUE
#     </parameter>...</function>`` (string-valued params, NOT JSON).
# Without recovery the run loop sees text + no tool_use and stalls
# ("went quiet" / "silent_finish"), which kills an entire family of
# open-weight coding models (qwen3-coder, hermes, devstral, ...). We
# recover these into real tool_uses, but ONLY as a fallback: see
# `parse_response` for the guards (no native tool_calls present AND the
# recovered name matches a tool that was actually offered). Flagship
# models that emit native tool_calls, and any model that legitimately
# answers with JSON, never hit this path.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# Qwen-Coder XML tool form. The closing `</function>` is sometimes
# missing (truncation) or mis-spelled `</tool_call>`; capture the name
# and a lenient body, then mine `<parameter=...>` pairs out of it.
# The next-tag terminators are LOOKAHEADS (not consuming): when a closing tag
# is missing, the body must end *before* the next block's opening tag without
# swallowing it -- otherwise finditer consumes that opener and silently drops
# the following function/parameter (corrupts e.g. apply_edit on open-weight
# models that emit unclosed Qwen-XML tool calls).
_FUNCTION_CALL_RE = re.compile(
    r"<function\s*=\s*([^>\s]+?)\s*>(.*?)(?:</function>|</tool_call>|(?=<function\s*=)|\Z)",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+?)\s*>(.*?)(?:</parameter>|(?=<parameter\s*=)|\Z)",
    re.DOTALL,
)
# Leftover scaffolding to scrub from the visible text once calls are mined.
# Orphan closers Qwen's template leaves right after a </function> block (a
# stray </tool_call> most commonly); swallowed into the recovered call's span.
_TRAILING_SCAFFOLD_RE = re.compile(r"(?:\s*(?:</tool_call>|</function>|</parameter>))+")

# Gemini / Gemma `tool_code` form: a fenced block of Python-call syntax, e.g.
#   ```tool_code
#   [read_file(path='spec.md'), apply_edit(path='x', ...)]
#   ```
# Gemma-family models on OpenRouter emit calls this way in `content` with empty
# native `tool_calls`, so without recovery the loop sees no tool_use and stops.
_TOOL_CODE_FENCE_RE = re.compile(r"```tool_code\s*\n?(.*?)```", re.DOTALL)


def lenient_json_object(raw: object) -> dict[str, Any] | None:
    """Recover a tool-call `arguments` string that strict `json.loads`
    rejected, when the fix is safe and unambiguous. Returns the object, or None.

    Two common weak/open-model malformations:
    - a raw control char (an unescaped newline/tab) inside a string value, which
      `strict=False` accepts;
    - trailing junk after a valid object (a leaked `</invoke>` tag or prose),
      which `raw_decode` ignores by parsing only the leading value.

    Only a dict result is returned; a scalar/array (or a still-invalid string,
    e.g. a bad `\\d` regex escape) yields None so the caller keeps the
    `_raw_arguments` sentinel rather than guessing."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed, _ = json.JSONDecoder(strict=False).raw_decode(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _tool_code_call_to_dict(node: ast.Call, tool_names: frozenset[str]) -> dict[str, Any] | None:
    """Turn one `ast.Call` into `{"name", "input"}` if it (or, unwrapping a
    non-tool wrapper such as `print(tool(...))`, an inner call) targets an
    offered tool. Keyword args are read with `ast.literal_eval` (already typed),
    so no coercion; non-literal or positional args are skipped. Returns None for a
    non-tool call -- we do NOT recurse into kwarg VALUES, so a tool nested as an
    argument (`apply_edit(path=read_file(...))`) is not separately mined."""
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id not in tool_names:
        # One-level unwrap: a non-tool wrapper around a single tool call.
        for arg in node.args:
            if isinstance(arg, ast.Call):
                inner = _tool_code_call_to_dict(arg, tool_names)
                if inner is not None:
                    return inner
        return None
    args: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:  # **kwargs splat -- not a named arg
            continue
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue  # a non-literal value (a name/expr); skip it
    return {"name": node.func.id, "input": args}


def _extract_tool_code_calls(
    text: str,
    tool_names: frozenset[str],
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Mine Gemini/Gemma ```tool_code Python-call blocks from leaked content.

    Parses each fenced block with `ast` (never executes it) and returns
    `([{"name", "input"}, ...], spans)` for every offered-tool call, in SOURCE
    ORDER; spans cover only fences that yielded calls. It walks only the TOP-LEVEL
    expressions (a bare call, or the elements of a list / tuple), unwrapping one
    `print(...)`-style wrapper -- not `ast.walk` (whose breadth-first order would
    reorder a tool call nested at a different depth)."""
    out: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for block in _TOOL_CODE_FENCE_RE.finditer(text):
        code = block.group(1).strip()
        if not code:
            continue
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError:
            continue
        calls_before = len(out)
        for stmt in tree.body:
            if not isinstance(stmt, ast.Expr):
                continue
            value = stmt.value
            elements = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            for el in elements:
                if isinstance(el, ast.Call):
                    call = _tool_code_call_to_dict(el, tool_names)
                    if call is not None:
                        out.append(call)
        if len(out) > calls_before:
            spans.append(block.span())
    return out, spans


def _coerce_param_value(value: str, declared_type: str | None) -> Any:  # noqa: PLR0911, PLR0912
    """Coerce a Qwen-XML `<parameter>` string to its schema-declared type.

    The Qwen-Coder template emits each parameter value as raw text framed by
    newlines, e.g. `<parameter=path>\\ninterp.py\\n</parameter>`. Strip the
    framing newlines, then coerce by the tool's declared JSON-Schema type so
    structured params (`array`/`object`) and scalars rebuild correctly
    while string params (code in `new_string`/`old_string`) are left byte-
    exact. Unknown type: parse only if it looks like JSON array/object, else
    keep the string.
    """
    # Strip the single leading/trailing newline the template adds without
    # touching interior or leading-space indentation that code params need.
    v = value
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    if declared_type == "string":
        return v
    if declared_type in ("array", "object"):
        try:
            return json.loads(v.strip())
        except (json.JSONDecodeError, TypeError):
            return v  # let pydantic surface a clear validation error
    if declared_type == "integer":
        try:
            return int(v.strip())
        except ValueError:
            return v
    if declared_type == "number":
        try:
            return float(v.strip())
        except ValueError:
            return v
    if declared_type == "boolean":
        normalized = v.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        return v
    # Unknown / absent schema: only auto-parse clearly-structured JSON so a
    # plain string value is never silently turned into a number or dict.
    stripped = v.strip()
    if stripped[:1] in ("[", "{"):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return v
    return v


def _extract_function_xml_calls(
    text: str,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Mine Qwen-Coder `<function=NAME><parameter=KEY>VALUE</parameter>`
    calls from leaked content text. Returns ``([{"name", "input"}, ...],
    spans)`` for every block whose name matches an offered tool (spans cover
    each recovered block plus its trailing orphan scaffold closers); empty
    if none match."""
    out: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for fmatch in _FUNCTION_CALL_RE.finditer(text):
        name = fmatch.group(1).strip()
        if name not in tool_names:
            continue
        body = fmatch.group(2)
        schema = (tool_schemas or {}).get(name) or {}
        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for pmatch in _PARAMETER_RE.finditer(body):
            key = pmatch.group(1).strip()
            decl = props.get(key) or {}
            decl_type = decl.get("type") if isinstance(decl, dict) else None
            args[key] = _coerce_param_value(pmatch.group(2), decl_type)
        out.append({"name": name, "input": args})
        # Extend the span over immediately-trailing orphan scaffold closers
        # (the template's stray </tool_call> etc.), so removing the recovered
        # call leaves no dangling tag behind.
        end = fmatch.end()
        trail = _TRAILING_SCAFFOLD_RE.match(text, end)
        if trail is not None:
            end = trail.end()
        spans.append((fmatch.start(), end))
    return out, spans


def _extract_tool_call_obj(  # noqa: PLR0911
    candidate: str, tool_names: frozenset[str]
) -> dict[str, Any] | None:
    """Parse a single `{"name", "arguments"}` tool call from a text
    candidate, or return None if it isn't a tool call for an offered tool."""
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    # Accept the common spellings local templates use for the args object.
    raw_args = obj.get("arguments")
    if raw_args is None:
        raw_args = obj.get("parameters")
    if raw_args is None:
        raw_args = obj.get("input")
    if raw_args is None:
        raw_args = {}
    # A few templates double-encode the args as a JSON string.
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(raw_args, dict):
        return None
    return {"name": name, "input": raw_args}


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """`text` with the given non-overlapping `(start, end)` spans cut out."""
    parts: list[str] = []
    prev = 0
    for start, end in spans:
        parts.append(text[prev:start])
        prev = end
    parts.append(text[prev:])
    return "".join(parts).strip()


def coerce_text_tool_calls(  # noqa: PLR0911
    text: str,
    tool_names: frozenset[str],
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Best-effort recovery of tool calls a local model emitted as text.

    Returns `(tool_uses, remaining_text)`. `tool_uses` is empty when
    nothing tool-call-shaped is found, in which case `remaining_text`
    equals the original `text`. The parsing is deliberately strict
    (exact JSON, a single fenced JSON object, `<tool_call>` tags, the
    `<function=...>` XML form, or a `tool_code` fence) so prose that
    merely mentions a tool name is never misread as a call.
    """
    if not text or not tool_names:
        return [], text
    # 0) Qwen-Coder `<function=NAME><parameter=KEY>VALUE</parameter></function>`
    # XML. Checked first: it is self-delimiting and unambiguous, and the
    # inner body is NOT JSON so the JSON-shaped branches below cannot parse it.
    xml_calls, xml_spans = _extract_function_xml_calls(text, tool_names, tool_schemas)
    if xml_calls:
        # Remove ONLY the calls that were recovered (same rule as the
        # <tool_call> branch below): a block whose name matched no offered
        # tool stays visible, so the model can see its failed call instead
        # of assuming it happened.
        return xml_calls, _remove_spans(text, xml_spans)
    # 0.5) Gemini / Gemma ```tool_code Python-call block. Self-delimiting like the
    # XML form, and parsed with ast (not JSON), so check it before the JSON
    # branches below.
    if "```tool_code" in text:
        code_calls, code_spans = _extract_tool_code_calls(text, tool_names)
        if code_calls:
            return code_calls, _remove_spans(text, code_spans)
    # 1) Hermes / Qwen `<tool_call>...</tool_call>` wrappers (≥1).
    tag_matches = list(_TOOL_CALL_TAG_RE.finditer(text))
    if tag_matches:
        recovered: list[dict[str, Any]] = []
        drop_spans: list[tuple[int, int]] = []
        for match in tag_matches:
            obj = _extract_tool_call_obj(match.group(1), tool_names)
            if obj is not None:
                recovered.append(obj)
                drop_spans.append(match.span())
        if recovered:
            # Remove ONLY the tags that parsed. A malformed sibling tag stays
            # in the remaining text so the model can see its failed call
            # (silently scrubbing it made the model assume the call happened).
            return recovered, _remove_spans(text, drop_spans)
    # 2) A single fenced JSON object that is itself a tool call.
    for fence in _JSON_FENCE_RE.finditer(text):
        obj = _extract_tool_call_obj(fence.group(1), tool_names)
        if obj is not None:
            # Remove only the matched fence; other ```json fences may be
            # legitimate content (a config sample, a reference block).
            return [obj], _remove_spans(text, [fence.span()])
    # 3) The whole content is exactly one bare JSON tool-call object.
    obj = _extract_tool_call_obj(text, tool_names)
    if obj is not None:
        return [obj], ""
    return [], text
