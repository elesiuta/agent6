# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the transcript -> conversation renderer (both provider wire shapes)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from agent6.paths import state_dir
from agent6.sessions.ipc import register_frontend
from agent6.viewmodel.transcript_render import (
    conversation_transcripts,
    fold_conversation,
    load_transcripts,
    render_markdown,
)

_OPENAI = [
    {
        "seq": 1,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "do X"},
                ]
            }
        },
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "working on it",
                            "reasoning_content": "let me think",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": "working on it",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "FULL FILE CONTENTS", "tool_call_id": "c1"},
                ]
            }
        },
        "response": {
            "body": {"choices": [{"message": {"role": "assistant", "content": "all done"}}]}
        },
    },
]

_ANTHROPIC = [
    {
        "seq": 1,
        "request": {
            "body": {
                "system": "SYSTEM PROMPT",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "do X"}]},
                ],
            }
        },
        "response": {
            "body": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "text", "text": "working on it"},
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "read_file",
                        "input": {"path": "a.py"},
                    },
                ],
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "system": "SYSTEM PROMPT",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "do X"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "u1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "u1",
                                "content": "FULL FILE CONTENTS",
                            }
                        ],
                    },
                ],
            }
        },
        "response": {
            "body": {"role": "assistant", "content": [{"type": "text", "text": "all done"}]}
        },
    },
]


_RESPONSES = [
    {
        "seq": 1,
        "request": {
            "body": {
                "instructions": "SYSTEM PROMPT",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "do X"}],
                    }
                ],
            }
        },
        "response": {
            "body": {
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "OPAQUE",
                        "summary": [{"type": "summary_text", "text": "let me think"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "working on it"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    },
                ],
                "status": "end_turn",
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "instructions": "SYSTEM PROMPT",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "do X"}],
                    },
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "encrypted_content": "OPAQUE",
                        "summary": [{"type": "summary_text", "text": "let me think"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "working on it"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read_file",
                        "arguments": '{"path":"a.py"}',
                    },
                    {"type": "function_call_output", "call_id": "c1", "output": "file body"},
                ],
            }
        },
        "response": {
            "body": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "status": "end_turn",
            }
        },
    },
]


def test_fold_and_render_the_responses_shape() -> None:
    """A ChatGPT (Responses) transcript: `instructions` is the system turn,
    `input` items fold into turns (one model response spans reasoning, a
    message and calls, so those items make ONE assistant turn), the next
    request's echo of the recorded output items is not printed twice, and a
    call output is labelled with its call's name."""
    turns = fold_conversation(_RESPONSES)
    assert [(t.role, t.seq) for t in turns] == [
        ("system", 1),
        ("user", 1),
        ("assistant", 1),
        ("tool", 2),
        ("assistant", 2),
    ]
    assert turns[0].text == "SYSTEM PROMPT" and turns[1].text == "do X"
    first = turns[2]
    assert first.thinking == "let me think" and first.text == "working on it"
    assert first.tool_calls == [("read_file", '{"path": "a.py"}')]
    assert turns[3].text == "file body" and turns[3].tool_name == "read_file"
    assert turns[4].text == "done"
    md = render_markdown(turns, session_id="s")
    assert "<thinking>\nlet me think\n</thinking>" in md
    assert md.count("working on it") == 1 and "-> read_file" in md


@pytest.mark.parametrize("transcripts", [_OPENAI, _ANTHROPIC], ids=["openai", "anthropic"])
def test_fold_and_render_both_shapes(transcripts: list[dict[str, Any]]) -> None:
    turns = fold_conversation(transcripts)
    roles = [t.role for t in turns]
    # system, user, assistant(seq1 w/ tool_call), tool result, assistant(seq2)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    a1 = turns[2]
    assert a1.text == "working on it" and a1.thinking == "let me think"
    assert a1.tool_calls and a1.tool_calls[0][0] == "read_file"
    assert "a.py" in a1.tool_calls[0][1]
    tool = turns[3]
    assert tool.tool_name == "read_file"  # resolved from the call id
    assert tool.text == "FULL FILE CONTENTS"  # full result, not a summary
    assert turns[4].text == "all done"

    md = render_markdown(turns, session_id="r1", show_thinking=True)
    assert "SYSTEM PROMPT" in md and "do X" in md and "all done" in md
    assert "-> read_file(" in md and "FULL FILE CONTENTS" in md
    assert "let me think" in md  # thinking shown


def test_render_flags_hide_thinking_and_tools() -> None:
    turns = fold_conversation(_OPENAI)
    md = render_markdown(turns, session_id="r1", show_thinking=False, tools="none")
    assert "let me think" not in md
    assert "-> read_file" not in md and "FULL FILE CONTENTS" not in md
    # calls-only keeps the call line but drops the result
    md2 = render_markdown(turns, session_id="r1", tools="calls")
    assert "-> read_file(" in md2 and "FULL FILE CONTENTS" not in md2


def test_provider_retry_does_not_duplicate_history() -> None:
    """A transient 5xx writes an error transcript (string body, no assistant
    turns) and the retry re-sends the IDENTICAL message list. The fold must
    treat that as no growth -- not as a compaction restart that prints a false
    'context summarised' marker and the entire history twice."""
    error_attempt = {
        "seq": 2,
        "request": _OPENAI[1]["request"],
        "response": {"status": 502, "body": "Bad Gateway"},
    }
    retry = {**_OPENAI[1], "seq": 3}
    turns = fold_conversation([_OPENAI[0], error_attempt, retry])
    assert [t.role for t in turns] == ["system", "user", "assistant", "tool", "assistant"]
    assert not any(t.role == "marker" for t in turns)


def test_compaction_restart_shows_marker() -> None:
    # seq 3's request is SHORTER than the prior history -> a summarise/restart.
    transcripts = [
        *_OPENAI,
        {
            "seq": 3,
            "request": {
                "body": {
                    "messages": [
                        {"role": "system", "content": "SYSTEM PROMPT"},
                        {"role": "user", "content": "<summary of earlier work>"},
                    ]
                }
            },
            "response": {
                "body": {"choices": [{"message": {"role": "assistant", "content": "resumed"}}]}
            },
        },
    ]
    turns = fold_conversation(transcripts)
    assert any(t.role == "marker" for t in turns)
    assert turns[-1].text == "resumed"


_BARE_ELIDED_A = (
    "<elided by context compaction: the result of read_file a.py was replaced"
    " with this short marker to keep the loop's cumulative input bounded. If"
    " you still need it, re-read only the part you need (read_file with a"
    " targeted offset/limit); do not re-issue the identical call.>"
)
_GIST_ELIDED_A = (
    "<elided by context compaction (distilled): the result of read_file a.py"
    " was replaced by this distilled gist; if the gist is not enough, re-read"
    " only the part you need (read_file with a targeted offset/limit).\n"
    "gist: R01 headers under 80 chars>"
)


def _anthropic_followup(elided_content: str) -> dict[str, Any]:
    """A seq-3 anthropic call: history grew by (assistant, user) AND the old
    tool_result at index 2 was mutated in place to an elision placeholder."""
    return {
        "seq": 3,
        "request": {
            "body": {
                "system": "SYSTEM PROMPT",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "do X"}]},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "u1",
                                "name": "read_file",
                                "input": {"path": "a.py"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "u1",
                                "content": elided_content,
                            }
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "all done"}]},
                    {"role": "user", "content": [{"type": "text", "text": "next step"}]},
                ],
            }
        },
        "response": {"body": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
    }


def test_tier1_elision_shows_marker_with_identity() -> None:
    """When a later request mutates an old tool_result into an elision
    placeholder, the conversation view says so instead of silently implying the
    model still sees the original."""
    turns = fold_conversation([*_ANTHROPIC, _anthropic_followup(_BARE_ELIDED_A)])
    markers = [t for t in turns if t.role == "marker"]
    assert len(markers) == 1
    assert "elided 1 older tool result" in markers[0].text
    assert "read_file a.py" in markers[0].text
    # The original result stays displayed; history is not re-printed.
    assert sum(1 for t in turns if t.text == "FULL FILE CONTENTS") == 1
    assert turns[-1].text == "ok"


def test_tier1_elision_marker_flags_kept_gists() -> None:
    turns = fold_conversation([*_ANTHROPIC, _anthropic_followup(_GIST_ELIDED_A)])
    markers = [t for t in turns if t.role == "marker"]
    assert len(markers) == 1
    assert "read_file a.py (distilled gist kept)" in markers[0].text


def test_tier1_elision_marker_openai_shape() -> None:
    followup = {
        "seq": 3,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM PROMPT"},
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": "working on it",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": _BARE_ELIDED_A, "tool_call_id": "c1"},
                    {"role": "assistant", "content": "all done"},
                    {"role": "user", "content": "next"},
                ]
            }
        },
        "response": {"body": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}},
    }
    turns = fold_conversation([*_OPENAI, followup])
    markers = [t for t in turns if t.role == "marker"]
    assert len(markers) == 1
    assert "elided 1 older tool result" in markers[0].text
    assert "read_file a.py" in markers[0].text


def test_gist_demotion_is_not_reported_as_a_fresh_elision() -> None:
    """A gist decaying to the bare marker changes the placeholder bytes, but the
    result was already reported elided -- no second 'elided' marker."""
    seq4 = _anthropic_followup(_BARE_ELIDED_A)
    seq4["seq"] = 4
    turns = fold_conversation([*_ANTHROPIC, _anthropic_followup(_GIST_ELIDED_A), seq4])
    markers = [t for t in turns if t.role == "marker"]
    assert len(markers) == 1  # the original gist elision only
    assert "distilled gist kept" in markers[0].text


def test_second_same_identity_elision_in_a_later_pass_is_counted() -> None:
    """Two results of the SAME call identity in one message, elided in two
    different passes: the second pass's marker must count the newly elided one
    (an identity SET would see it as already-reported and under-count)."""
    both = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": "FULL A"},
            {"type": "tool_result", "tool_use_id": "u2", "content": "FULL A COPY"},
        ],
    }
    assistant = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "u1", "name": "read_file", "input": {"path": "a.py"}},
            {"type": "tool_use", "id": "u2", "name": "read_file", "input": {"path": "a.py"}},
        ],
    }
    task = {"role": "user", "content": [{"type": "text", "text": "do X"}]}

    def call(seq: int, msgs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "seq": seq,
            "request": {"body": {"system": "S", "messages": msgs}},
            "response": {
                "body": {"role": "assistant", "content": [{"type": "text", "text": f"t{seq}"}]}
            },
        }

    one_elided = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": _BARE_ELIDED_A},
            {"type": "tool_result", "tool_use_id": "u2", "content": "FULL A COPY"},
        ],
    }
    both_elided = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "u1", "content": _BARE_ELIDED_A},
            {"type": "tool_result", "tool_use_id": "u2", "content": _BARE_ELIDED_A},
        ],
    }
    grow = {"role": "user", "content": [{"type": "text", "text": "next"}]}
    turns = fold_conversation(
        [
            call(1, [task, assistant, both]),
            call(2, [task, assistant, one_elided, grow]),
            call(3, [task, assistant, both_elided, grow, grow]),
        ]
    )
    markers = [t.text for t in turns if t.role == "marker"]
    assert len(markers) == 2
    assert "elided 1 older tool result" in markers[0]
    assert "elided 1 older tool result" in markers[1]  # the second copy, counted


def test_elision_marker_prefix_matches_the_compaction_placeholder() -> None:
    """The renderer detects placeholders by prefix; this pins the cross-module
    coupling without a runtime viewmodel->workflows import."""
    from agent6.viewmodel.transcript_render import ELISION_MARKER_PREFIX
    from agent6.workflows._compaction import ELISION_PREFIX

    assert ELISION_MARKER_PREFIX == ELISION_PREFIX


def test_load_transcripts_sorted_by_seq(tmp_path: Path) -> None:
    d = tmp_path / "transcripts"
    d.mkdir()
    (d / "20260101T2-000002.json").write_text(json.dumps({"seq": 2, "x": "b"}), encoding="utf-8")
    (d / "20260101T1-000001.json").write_text(json.dumps({"seq": 1, "x": "a"}), encoding="utf-8")
    (d / "bad.json").write_text("{not json", encoding="utf-8")  # skipped, not fatal
    loaded = load_transcripts(d)
    assert [t["seq"] for t in loaded] == [1, 2]
    assert load_transcripts(tmp_path / "nope") == []


def test_cmd_history_transcript_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 sessions transcript <run>` resolves the run, folds its transcripts,
    and prints the conversation (full tool I/O), with --json as the raw escape."""
    from agent6.ui.cli.history_cmds import (
        _cmd_history_transcript,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "st"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    tdir = state_dir(repo) / "sessions" / "runs" / "my-run" / "transcripts"
    tdir.mkdir(parents=True)
    (tdir.parent / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    (tdir / "20260101-000001.json").write_text(json.dumps(_OPENAI[0]), encoding="utf-8")
    (tdir / "20260101-000002.json").write_text(json.dumps(_OPENAI[1]), encoding="utf-8")

    rc = _cmd_history_transcript("my-run", as_json=False, no_thinking=False, tools="both", seq="")
    assert rc == 0
    out = capsys.readouterr().out
    assert "Transcript: my-run" in out and "-> read_file(" in out and "FULL FILE CONTENTS" in out

    rc_json = _cmd_history_transcript(
        "my-run", as_json=True, no_thinking=False, tools="both", seq="2"
    )
    assert rc_json == 0
    data = json.loads(capsys.readouterr().out)
    assert [t["seq"] for t in data] == [2]  # --seq windowed the raw transcripts

    assert (
        _cmd_history_transcript("nope", as_json=False, no_thinking=False, tools="both", seq="") == 2
    )


def test_cmd_history_transcript_latest_uses_log_activity_not_dir_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli.history_cmds import (
        _cmd_history_transcript,  # pyright: ignore[reportPrivateUsage]
    )

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "st"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    runs = state_dir(repo) / "sessions" / "runs"
    for name in ("older-run", "newer-run"):
        tdir = runs / name / "transcripts"
        tdir.mkdir(parents=True)
        (tdir / "20260101-000001.json").write_text(json.dumps(_OPENAI[0]), encoding="utf-8")
        (runs / name / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(runs / "older-run" / "logs.jsonl", (100, 100))
    os.utime(runs / "newer-run" / "logs.jsonl", (1000, 1000))
    register_frontend(runs / "older-run", 12345)

    assert _cmd_history_transcript("", as_json=True, no_thinking=False, tools="both", seq="") == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)[0]["seq"] == 1
    assert "newer-run" in captured.err


def test_streamed_openai_response_without_a_role_is_the_assistant() -> None:
    """The streaming path synthesises choices[0].message with no "role" key (a
    real OpenAI response always carries one), and that body is what the recorder
    writes. Rendering it fell through to the generic branch, so the model's words
    printed under '## user', tool_calls were dropped, reasoning was lost, and the
    unresolved call id left every later result unlabelled."""
    streamed = [
        {
            "seq": 1,
            "request": {"body": {"messages": [{"role": "user", "content": "do X"}]}},
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {  # no "role" -- exactly what _call_streaming records
                                "content": "working on it",
                                "reasoning_content": "let me think",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"a.py"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            },
        },
        {
            "seq": 2,
            "request": {
                "body": {
                    "messages": [
                        {"role": "user", "content": "do X"},
                        {
                            "role": "assistant",
                            "content": "working on it",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a.py"}',
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "c1", "content": "FULL FILE CONTENTS"},
                    ]
                }
            },
            "response": {"body": {"choices": [{"message": {"content": "all done"}}]}},
        },
    ]
    turns = fold_conversation(streamed)
    assert [t.role for t in turns] == ["user", "assistant", "tool", "assistant"]
    a1 = turns[1]
    assert a1.thinking == "let me think"
    assert a1.tool_calls and a1.tool_calls[0][0] == "read_file"
    assert turns[2].tool_name == "read_file"  # the call id resolved to a name
    md = render_markdown(turns, session_id="r1", show_thinking=True)
    assert "## assistant" in md and "-> read_file(" in md


def test_a_compaction_side_call_is_not_a_conversation_turn(tmp_path: Path) -> None:
    """The gist distiller and the tier-2 summariser share the run's transcript
    sink, and their ONE-message requests shrink the history, which the fold reads
    as a compaction restart: it printed a phantom "context summarised" marker,
    rendered the side-call's scratch prompt as a user turn, and re-emitted the
    history behind it. Only the worker seat is the conversation."""
    import json

    d = tmp_path / "transcripts"
    d.mkdir()

    def rec(seq: int, seat: str, msgs: list[dict[str, object]], reply: str) -> None:
        (d / f"2026-{seq:06d}.json").write_text(
            json.dumps(
                {
                    "seq": seq,
                    "seat": seat,
                    "request": {"url": "", "headers": {}, "body": {"messages": msgs}},
                    "response": {
                        "status": 200,
                        "body": {"content": [{"type": "text", "text": reply}]},
                    },
                }
            ),
            encoding="utf-8",
        )

    def u(text: str) -> dict[str, object]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def a(text: str) -> dict[str, object]:
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    rec(1, "worker", [u("TASK: fix the parser"), a("reading"), u("tool result")], "on it")
    rec(2, "reviewer", [u("=== FILE a.py ===")], "a.py: the parser")  # the distiller
    rec(
        3,
        "worker",
        [u("TASK: fix the parser"), a("reading"), u("tool result"), a("on it"), u("next")],
        "done",
    )

    turns = fold_conversation(load_transcripts(d))
    body = "\n".join(f"{t.role}:{t.text}" for t in turns)
    assert "context summarised" not in body, "a restart that never happened"
    assert "=== FILE a.py ===" not in body, "the side-call's scratch prompt became a turn"
    assert body.count("reading") == 1, "the history was re-emitted behind the side-call"


def test_the_conversation_seat_is_the_driving_provider_not_always_worker(tmp_path: Path) -> None:
    """The seat filter first kept only "worker", but the loop's driving provider
    takes its role from the mode: plan mode's is "planner", so every plan run's
    transcripts were filtered out and `history transcript` said the run had none
    while the files sat on disk. Review seats must still be excluded -- they
    share the run's sink and their one-message requests read as a restart."""
    import json

    d = tmp_path / "transcripts"
    d.mkdir()
    for seq, seat in ((1, "planner"), (2, "reviewer"), (3, "review:security"), (4, "")):
        (d / f"{seq:06d}.json").write_text(
            json.dumps(
                {
                    "seq": seq,
                    "seat": seat,
                    "request": {"url": "", "headers": {}, "body": {"messages": []}},
                    "response": {"status": 200, "body": {}},
                }
            ),
            encoding="utf-8",
        )

    kept = [t["seat"] for t in conversation_transcripts(load_transcripts(d))]
    assert "planner" in kept, "a plan run lost its whole conversation"
    assert "" in kept, "a transcript written before seats existed is the driving seat's"
    assert "reviewer" not in kept and "review:security" not in kept


def test_load_transcripts_stays_raw_for_the_json_dump(tmp_path: Path) -> None:
    """`sessions transcript --json` advertises "the raw transcript array", and it is
    the one CLI surface for a side-call's actual request/response (the thing you
    need to debug a bad compaction). The seat filter lives in the CONVERSATION
    fold, not the loader, so the dump keeps every seat."""
    import json

    d = tmp_path / "transcripts"
    d.mkdir()
    for seq, seat in ((1, "worker"), (2, "reviewer"), (3, "review:security")):
        (d / f"{seq:06d}.json").write_text(
            json.dumps(
                {
                    "seq": seq,
                    "seat": seat,
                    "request": {"url": "", "headers": {}, "body": {"messages": []}},
                    "response": {"status": 200, "body": {}},
                }
            ),
            encoding="utf-8",
        )
    assert [t["seat"] for t in load_transcripts(d)] == ["worker", "reviewer", "review:security"]


def test_a_replayed_message_stripped_of_its_id_is_not_printed_twice() -> None:
    """The ChatGPT wire records a message item with an id, status, phase and
    content annotations, and replays it without them; the fold compared the
    two dicts whole, so every assistant message printed twice (once from its
    response, once from the next request's echo)."""
    import json

    transcripts = json.loads(json.dumps(_RESPONSES))
    recorded = transcripts[0]["response"]["body"]["output"][1]
    recorded.update({"id": "msg_1", "status": "completed", "phase": "commentary"})
    recorded["content"][0].update({"annotations": [], "logprobs": []})
    turns = fold_conversation(transcripts)
    assert [(t.role, t.seq) for t in turns] == [
        ("system", 1),
        ("user", 1),
        ("assistant", 1),
        ("tool", 2),
        ("assistant", 2),
    ]
    assert render_markdown(turns, session_id="s").count("working on it") == 1


def test_a_transcript_whose_seq_is_not_a_number_is_kept_and_marked(tmp_path: Path) -> None:
    """A `seq` that is not an integer crashed the sort, the fold and the
    `--seq` window (a bare TypeError, then a ValueError). It orders as 0, the
    record still reaches `sessions transcript --json` verbatim, and the fold
    says what it could not place."""
    from agent6.viewmodel.transcript_render import fold_conversation, load_transcripts

    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "0000001.json").write_text(
        json.dumps({"seq": "notanumber", "seat": "worker", "request": {}, "response": {}}),
        encoding="utf-8",
    )
    (tdir / "0000002.json").write_text(
        json.dumps({"seq": 2, "seat": "worker", "request": {}, "response": {}}),
        encoding="utf-8",
    )
    loaded = load_transcripts(tdir)
    assert [t["seq"] for t in loaded] == ["notanumber", 2]
    marks = [t.text for t in fold_conversation(loaded) if t.role == "marker"]
    assert any("unreadable seq 'notanumber'" in m for m in marks)
