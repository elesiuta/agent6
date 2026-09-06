# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.tools.dispatch — path safety, edit semantics, no-net I/O."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from agent6.config import Config
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.operator_prompts import (
    ApprovalAnswer,
    ApprovalRequest,
    OperatorPrompts,
    QuestionAnswer,
    QuestionRequest,
)
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
prompt_caching = true
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[sandbox]
isolation = "auto"
run_commands = "no"
protect_git = true
[git]
dirty_tree = "ask"
branch_per_run = true
[workflow]
verify_command = ["true"]
[budget]
max_tokens_fallback = 2000000
"""


def _config(tmp_path: Path) -> Config:
    from agent6.config import load_config

    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _config_with_run_commands(tmp_path: Path, mode: str) -> Config:
    from agent6.config import load_config

    p = tmp_path / f"agent6-{mode}.toml"
    p.write_text(
        _VALID_TOML.replace('run_commands = "no"', f'run_commands = "{mode}"'),
        encoding="utf-8",
    )
    return load_config(p)


def test_read_file_ok(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("read_file", {"path": "hello.txt"}).to_wire()
    assert out["content"] == "hi"


def test_verify_command_unexecutable_raises_loud(tmp_path: Path) -> None:
    """A verify_command that the jail could not execute (rc 127, exec_failed)
    must raise OperatorCommandUnexecutable, not return a silent verify-failure.

    Regression: on a no-userns host the jail PATH is /usr/bin:/bin; a uv-based
    verify (uv lives under /usr/local/bin or ~/.local/bin) exited 127 and was
    reported as an ordinary verify failure (ok=True, exit=127), so the run
    reported all_passed and committed unverified work. The model cannot fix
    operator config, so this must fail loudly instead.
    """
    from agent6.tools.dispatch import OperatorCommandUnexecutable
    from agent6.types import CommandResult

    # run_commands = "yes": this exercises verify EXECUTION, not the gate.
    cfg = _config_with_run_commands(tmp_path, "yes")  # verify_command = ["true"]
    d = ToolDispatcher(root=tmp_path, config=cfg)
    unexecutable = CommandResult(
        argv=("true",),
        returncode=127,
        stdout="",
        stderr="true: command not found or not executable (agent6-jail: child execution failed)",
        duration_s=0.0,
        exec_failed=True,
    )
    with (
        mock.patch("agent6.tools.dispatch.run_in_jail", return_value=unexecutable),
        pytest.raises(OperatorCommandUnexecutable),
    ):
        d.dispatch("run_verify_command", {})

    # An ordinary non-zero exit (ran but failed) must NOT raise -- it is a real
    # verify failure the model can act on.
    ran_and_failed = CommandResult(
        argv=("true",), returncode=1, stdout="", stderr="assert", duration_s=0.1, exec_failed=False
    )
    with mock.patch("agent6.tools.dispatch.run_in_jail", return_value=ran_and_failed):
        out = d.dispatch("run_verify_command", {}).to_wire()
    assert out["returncode"] == 1


def test_apply_edit_tolerates_a_uniform_indent_mismatch(tmp_path: Path) -> None:
    # The dominant weak-model miss: right lines, wrong indent depth. old_string
    # is written at base indent 4; the file uses 8. The edit applies, and the
    # result keeps the FILE's indentation (not the model's).
    cfg = _config(tmp_path)
    (tmp_path / "m.py").write_text(
        "class C:\n    def m(self):\n        x = 1\n        return x\n", encoding="utf-8"
    )
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "m.py",
            "edits": [
                {
                    "kind": "replace",
                    "old_string": "def m(self):\n    x = 1\n    return x",
                    "new_string": "def m(self):\n    x = 2\n    return x",
                }
            ],
        },
    )
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == (
        "class C:\n    def m(self):\n        x = 2\n        return x\n"
    )


def test_apply_edit_refuses_an_ambiguous_indent_match(tmp_path: Path) -> None:
    # A multi-line block that matches TWO regions up to indent (0 exact matches):
    # never guess; surface the mismatch error, don't fuzzy-apply.
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text(
        "if x:\n    a = 1\n    b = 2\nif y:\n        a = 1\n        b = 2\n", encoding="utf-8"
    )
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not found"):
        d.dispatch(
            "apply_edit",
            {
                "path": "a.py",
                "edits": [{"old_string": "a = 1\nb = 2", "new_string": "a = 9\nb = 2"}],
            },
        )
    # Unchanged on disk (no guess applied).
    assert "a = 9" not in (tmp_path / "a.py").read_text(encoding="utf-8")


def test_raw_arguments_sentinel_gives_a_clear_json_error(tmp_path: Path) -> None:
    # When the provider couldn't parse the tool-call arguments as JSON it leaves
    # the {"_raw_arguments": ...} sentinel; dispatch must tell the model plainly
    # the JSON was malformed (not a confusing "extra fields" schema error) so it
    # resends in one shot.
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with pytest.raises(ToolError, match="not valid JSON"):
        d.dispatch("read_file", {"_raw_arguments": '{"path": "a.py"'})


def test_runaway_raw_arguments_name_the_truncation(tmp_path: Path) -> None:
    # A huge unterminated argument string is a runaway generation cut off by
    # the output-token ceiling (observed: a model emitting a 117KB argument of
    # one alternation repeated). "Resend" feedback makes such a model
    # regenerate the same runaway; the error must name the truncation and
    # direct a much smaller call instead.
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    runaway = '{"path": "' + "setup_show|" * 3000
    with pytest.raises(ToolError, match="cut off mid-generation"):
        d.dispatch("read_file", {"_raw_arguments": runaway})


def test_ask_user_routes_to_questioner(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    seen: dict[str, object] = {}

    def questioner(request: QuestionRequest, /) -> QuestionAnswer:
        seen["questions"] = request.questions
        answers = tuple(q.options[1] if q.options else "typed" for q in request.questions)
        return QuestionAnswer(answers, "stdin")

    d = ToolDispatcher(root=tmp_path, config=cfg, prompts=OperatorPrompts(questioner=questioner))
    out = d.dispatch(
        "ask_user", {"questions": [{"question": "which?", "options": ["a", "b"]}]}
    ).to_wire()
    assert out == {"answers": ["b"]}
    assert seen["questions"] == (UserQuestion(question="which?", options=("a", "b")),)


def test_default_questioner_headless_returns_empty(tmp_path: Path) -> None:
    # No injected questioner + EOF on stdin (headless) -> empty answers, no hang.
    from agent6.tools.operator_prompts import (
        _default_questioner,  # pyright: ignore[reportPrivateUsage]
    )

    request = QuestionRequest(
        id="question-1", questions=(UserQuestion(question="q?", options=("a", "b")),), call_id=1
    )
    with mock.patch("builtins.input", side_effect=EOFError):
        assert _default_questioner(request).answers == ("",)


def test_ask_user_refused_outside_run_mode(tmp_path: Path) -> None:
    # ask_user is a run-mode tool; the dispatcher backstops it in other modes
    # so a tool-list regression can't pause a plan/ask/machine loop.
    cfg = _config(tmp_path)
    prompts = OperatorPrompts(questioner=lambda request: QuestionAnswer(("x",), "stdin"))
    d = ToolDispatcher(root=tmp_path, config=cfg, mode="plan", prompts=prompts)
    with pytest.raises(ToolError, match="not available in plan mode"):
        d.dispatch("ask_user", {"questions": [{"question": "q?"}]})


@pytest.mark.parametrize("mode", ["plan", "ask", "machine"])
def test_run_metric_refused_outside_run_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """run_metric_command executes the operator's metric command with no
    approval gate; it is run-only (LOOP_EXTRA_TOOLS) and the dispatcher must
    backstop it in every other mode even with [workflow.metric] configured."""
    body = _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"') + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"CYCLES: 42\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config

    cfg = load_config(p)
    fired: list[object] = []

    def fake_run_in_jail(policy: object, **_kw: object) -> None:
        fired.append(policy)

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg, mode=mode)  # type: ignore[arg-type]
    with pytest.raises(ToolError, match=f"not available in {mode} mode"):
        d.dispatch("run_metric_command", {})
    assert fired == []


def test_mode_backstop_is_the_mode_tool_surface(tmp_path: Path) -> None:
    """The backstop derives from the same per-mode surface tool_definitions
    exposes, so a control tool outside a mode's surface is refused without
    needing its own hand-list entry (finish_planning in run and run_metric
    outside run were the gaps the hand-lists missed). agent6_docs is the one
    exemption: exposed only in ask, but a read-only doc fetch the review seat
    dispatches in every mode, so it stays permitted everywhere."""
    cfg = _config(tmp_path)
    run = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not available in run mode"):
        run.dispatch("finish_planning", {})
    assert run.dispatch("agent6_docs", {}).to_wire()["available"]
    machine = ToolDispatcher(root=tmp_path, config=cfg, mode="machine")
    assert machine.dispatch("agent6_docs", {}).to_wire()["available"]


def test_absolute_path_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch("read_file", {"path": "/etc/passwd"})


def test_parent_traversal_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.\."):
        d.dispatch("read_file", {"path": "../outside.txt"})


def test_read_file_allows_nested_dotdir(tmp_path: Path) -> None:
    # A leading-dot path component does not block reads; .agent6/ is no longer
    # special (run state lives out of the repo).
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    target = tmp_path / ".cache" / "foo"
    target.mkdir(parents=True)
    (target / "x.jsonl").write_text("{}\n", encoding="utf-8")
    out = d.dispatch("read_file", {"path": ".cache/foo/x.jsonl"}).to_wire()
    assert out["content"] == "{}\n"


def test_apply_edit_refuses_git_dir(tmp_path: Path) -> None:
    # apply_edit writes in-process (outside the jail), so without a guard the
    # LLM could plant a .git/hooks/pre-commit or rewrite .git/config and get
    # code run outside the sandbox on the next commit -- bypassing protect_git.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_edit",
            {
                "path": ".git/hooks/pre-commit",
                "edits": [{"kind": "create", "old_string": "", "new_string": "#!/bin/sh\nid\n"}],
            },
        )


def test_apply_patch_refuses_git_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_patch",
            {
                "path": ".git/config",
                "patch": "--- /dev/null\n+++ .git/config\n@@ -0,0 +1 @@\n+[core]\n",
            },
        )


def test_apply_edit_refuses_git_via_symlink(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / ".git").mkdir()
    (tmp_path / "decoy").symlink_to(".git", target_is_directory=True)
    with pytest.raises(ToolError, match=r"\.git.*symlink"):
        d.dispatch(
            "apply_edit",
            {
                "path": "decoy/hooks/pre-commit",
                "edits": [{"kind": "create", "old_string": "", "new_string": "x"}],
            },
        )


def test_apply_edit_allows_nested_git_dir(tmp_path: Path) -> None:
    """Allowed by the operator's ruling, not by oversight: `protect_git` covers
    the project's own repository, the one agent6 commits to each turn. A nested
    `.git` (vendored repo, submodule gitlink) is content like any other file --
    tracked by the root repo or untracked, either way no guarantee is offered
    over it. Holds raw and symlink-resolved."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "vendor/dep/.git/config",
            "edits": [{"kind": "create", "old_string": "", "new_string": "[core]\n"}],
        },
    )
    assert (tmp_path / "vendor/dep/.git/config").read_text(encoding="utf-8") == "[core]\n"
    # A submodule gitlink is a FILE named .git.
    d.dispatch(
        "apply_edit",
        {
            "path": "sub/.git",
            "edits": [{"kind": "create", "old_string": "", "new_string": "gitdir: x\n"}],
        },
    )
    assert (tmp_path / "sub" / ".git").read_text(encoding="utf-8") == "gitdir: x\n"
    (tmp_path / "innocent").symlink_to(Path("vendor/dep/.git"), target_is_directory=True)
    d.dispatch(
        "apply_edit",
        {
            "path": "innocent/hooks/pre-commit",
            "edits": [{"kind": "create", "old_string": "", "new_string": "#!/bin/sh\n"}],
        },
    )
    assert (tmp_path / "vendor/dep/.git/hooks/pre-commit").exists()


def test_apply_edit_allows_git_lookalike_components(tmp_path: Path) -> None:
    # Only the exact top-level component `.git` is protected: `git`,
    # `.gitignore`, `.github`, and any other `.git*` name stay writable.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    for path in (
        "src/git/main.py",
        "a/.gitignore",
        ".github/workflows/ci.yml",
        "vendor/.gitlab/ci.yml",
    ):
        d.dispatch(
            "apply_edit",
            {
                "path": path,
                "edits": [{"kind": "create", "old_string": "", "new_string": "ok\n"}],
            },
        )
        assert (tmp_path / path).read_text(encoding="utf-8") == "ok\n"


def test_apply_edit_allows_git_write_when_protect_git_false(tmp_path: Path) -> None:
    # Opting out of protect_git lifts the guard (consistent with the jail,
    # which also stops RO-binding .git when protect_git is false).
    from agent6.config import load_config

    p = tmp_path / "agent6-nogit.toml"
    p.write_text(_VALID_TOML.replace("protect_git = true", "protect_git = false"), encoding="utf-8")
    cfg = load_config(p)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": ".git/description",
            "edits": [{"kind": "create", "old_string": "", "new_string": "ok\n"}],
        },
    )
    assert (tmp_path / ".git" / "description").read_text(encoding="utf-8") == "ok\n"


def test_apply_edit_refuses_writes_inside_a_virtualenv(tmp_path: Path) -> None:
    # A run rewriting an editable-install .pth inside .venv to make an in-jail
    # verify pass silently corrupts the operator's environment (venvs are
    # gitignored, so it never shows in the run's diff). pyvenv.cfg marks the
    # venv root; the name is irrelevant.
    cfg = _config(tmp_path)
    venv = tmp_path / ".venv"
    site = venv / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    pth = site / "_editable_impl_pkg.pth"
    pth.write_text("/home/user/proj/src\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"virtualenv|site-packages"):
        d.dispatch(
            "apply_edit",
            {
                "path": ".venv/lib/python3.14/site-packages/_editable_impl_pkg.pth",
                "edits": [
                    {
                        "kind": "replace",
                        "old_string": "/home/user/proj/src",
                        "new_string": "/workspace/src",
                    }
                ],
            },
        )
    assert pth.read_text(encoding="utf-8") == "/home/user/proj/src\n"  # untouched


def test_apply_edit_refuses_site_packages_outside_a_pyvenv(tmp_path: Path) -> None:
    # A site-packages tree without a pyvenv.cfg above it (a bare install layout)
    # is still installed environment, not source.
    cfg = _config(tmp_path)
    site = tmp_path / "env" / "site-packages" / "pkg"
    site.mkdir(parents=True)
    (site / "mod.py").write_text("x = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="site-packages"):
        d.dispatch(
            "apply_edit",
            {
                "path": "env/site-packages/pkg/mod.py",
                "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 2"}],
            },
        )


def test_apply_edit_allows_a_normal_source_file_named_like_env(tmp_path: Path) -> None:
    # The guard keys on pyvenv.cfg / a site-packages ancestor, not on a name:
    # a source file under a dir merely called "env" (no pyvenv.cfg) is fine.
    cfg = _config(tmp_path)
    src = tmp_path / "env" / "settings.py"
    src.parent.mkdir()
    src.write_text("DEBUG = False\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "env/settings.py",
            "edits": [{"kind": "replace", "old_string": "False", "new_string": "True"}],
        },
    )
    assert "DEBUG = True" in src.read_text(encoding="utf-8")


def test_apply_edit_refuses_extra_protect_paths(tmp_path: Path) -> None:
    # A machine bundle's .asm.toml + scripts/ are passed as extra_protect_paths.
    # The jail marks them read-only for run_command, but the in-process edit tools
    # ran outside that -- a mode="run" state could rewrite its own logic/scripts
    # and persist a payload for the next run. Refuse on both isolation levels.
    cfg = _config(tmp_path)
    (tmp_path / "bundle").mkdir()
    asm = tmp_path / "bundle" / "fixer.asm.toml"
    asm.write_text("[machine]\n", encoding="utf-8")
    scripts = tmp_path / "bundle" / "scripts"
    scripts.mkdir()
    (scripts / "deploy.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    d = ToolDispatcher(
        root=tmp_path,
        config=cfg,
        extra_protect_paths=(asm.resolve(), scripts.resolve()),
    )
    with pytest.raises(ToolError, match="protected path"):
        d.dispatch(
            "apply_edit",
            {
                "path": "bundle/fixer.asm.toml",
                "edits": [{"kind": "replace", "old_string": "[machine]", "new_string": "[m]"}],
            },
        )
    with pytest.raises(ToolError, match="protected path"):
        d.dispatch(
            "apply_edit",
            {
                "path": "bundle/scripts/deploy.sh",
                "edits": [{"kind": "replace", "old_string": "true", "new_string": "curl evil"}],
            },
        )
    # A sibling file NOT under a protect path stays editable.
    (tmp_path / "bundle" / "notes.md").write_text("hi\n", encoding="utf-8")
    d.dispatch(
        "apply_edit",
        {
            "path": "bundle/notes.md",
            "edits": [{"kind": "replace", "old_string": "hi", "new_string": "bye"}],
        },
    )


def test_apply_edit_rejects_create_combined_with_other_edits(tmp_path: Path) -> None:
    # A `create` after a `replace` used to skip the file-exists guard (which
    # only fired for the first edit) and silently overwrite the whole file.
    # The schema now requires create to be the sole edit.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("keep me\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="create"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [
                    {"kind": "replace", "old_string": "keep me", "new_string": "edited"},
                    {"kind": "create", "old_string": "", "new_string": "OVERWRITE\n"},
                ],
            },
        )
    # File untouched (the call was rejected before any write).
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "keep me\n"


def test_apply_edit_create_and_replace(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "x = 1\n"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 2"}],
        },
    )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 2\n"


def test_apply_edit_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """Both edit tools create a file whose directories do not exist yet; the
    containment walk creates them, so an ordinary new package still lands."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {
            "path": "pkg/sub/mod.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "x = 1\n"}],
        },
    )
    assert (tmp_path / "pkg" / "sub" / "mod.py").read_text(encoding="utf-8") == "x = 1\n"
    d.dispatch(
        "apply_patch",
        {
            "path": "pkg/other/new.py",
            "patch": "--- /dev/null\n+++ b/pkg/other/new.py\n@@ -0,0 +1 @@\n+y = 2\n",
        },
    )
    assert (tmp_path / "pkg" / "other" / "new.py").read_text(encoding="utf-8") == "y = 2\n"


def test_apply_edit_non_unique_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\na\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not unique"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "a", "new_string": "b"}],
            },
        )


def test_apply_edit_missing_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not found"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "bye", "new_string": "x"}],
            },
        )


def test_apply_edit_not_found_error_format(tmp_path: Path) -> None:
    # Finding C: the "old_string not found" error must NOT wrap the
    # file body in `---BEGIN <path>---` / `---END <path>---` markers, and
    # must NOT dump the entire body. Models that degenerate on repetition
    # were observed copying the marker scaffolding verbatim into the next
    # old_string. The new format gives "shape" (size, line count, head,
    # tail) and a tells-the-worker-what-to-do recovery hint.
    cfg = _config(tmp_path)
    body = "\n".join(f"line {i}" for i in range(1, 21)) + "\n"  # 20 lines
    (tmp_path / "f.py").write_text(body, encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc_info:
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [
                    {"kind": "replace", "old_string": "no such text here", "new_string": "x"}
                ],
            },
        )
    msg = str(exc_info.value)
    assert "old_string not found" in msg
    # No scaffolding markers — these were the leak-back vector.
    assert "---BEGIN" not in msg
    assert "---END" not in msg
    # Shape and tail markers are present.
    assert f"{len(body)} bytes" in msg
    assert "20 lines" in msg
    assert "first 5 lines:" in msg
    assert "last 5 lines:" in msg
    # The full file body MUST NOT be in the error (otherwise the model
    # might still echo it back wholesale).
    assert "line 10" not in msg
    # Recovery hint must tell the model what to do next.
    assert "read_file" in msg


def test_apply_edit_not_found_short_file_omits_tail(tmp_path: Path) -> None:
    # Files of <=10 lines don't need the "...last 5 lines" duplication.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc_info:
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "no match", "new_string": "x"}],
            },
        )
    msg = str(exc_info.value)
    assert "first 5 lines:" in msg
    assert "last 5 lines:" not in msg
    assert "3 lines" in msg


def test_apply_edit_replace_requires_new_string(tmp_path: Path) -> None:
    # Kimi was emitting {kind:"replace", old_string:"..."}
    # WITHOUT a new_string. The old default `new_string: str = ""` silently
    # turned a malformed replace into a deletion, which corrupted the file
    # and then put the agent into an unrecoverable hallucination loop. The
    # boundary now rejects the malformed input loud per AGENTS.md.
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="new_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "x = 1"}],
            },
        )
    # File untouched.
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_edit_replace_rejects_empty_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("hello\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="old_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "replace", "old_string": "", "new_string": "x"}],
            },
        )


def test_invalid_arguments_read_as_one_line(tmp_path: Path) -> None:
    """A schema miss is the model's most common recoverable error; it gets one
    line naming the field and the rule, not pydantic's dump (two errors, docs
    URLs, and a cascade "Tuple should have at least 1 item" for the container
    whose only item was invalid)."""
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as caught:
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "replace", "old_string": "", "new_string": "y\n"}]},
        )
    message = str(caught.value)
    assert message.startswith(
        "invalid arguments: edits.0: old_string must be non-empty for kind='replace'"
    )
    # The caller wanted to write a file; the refusal names the kind that does
    # it. Without `create`, a model ping-pongs between replace and overwrite
    # until the tool-error streak stops the run (seen live in `machine create`).
    assert "kind='create'" in message
    assert "\n" not in str(caught.value)
    with pytest.raises(ToolError, match=r"^invalid arguments: edits: Field required$"):
        d.dispatch("apply_edit", {"path": "f.py"})


def test_apply_edit_create_rejects_nonempty_old_string(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="old_string"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "create", "old_string": "junk", "new_string": "x"}],
            },
        )


def test_apply_edit_overwrite_replaces_an_existing_file_whole(tmp_path: Path) -> None:
    """A rewrite from a stub: `create` refuses the existing file (and names
    the kind that does the job); `overwrite` writes it whole and reports so."""
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("def f():\n    raise NotImplementedError\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"already exists.*overwrite"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "create", "old_string": "", "new_string": "x"}]},
        )
    out = d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [
                {"kind": "overwrite", "old_string": "", "new_string": "def f():\n    return 1\n"}
            ],
        },
    ).to_wire()
    assert out["applied"] == ["overwrite"]
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"
    # A missing file is written the same way: the kind states "the whole
    # file is new_string", not "the file exists".
    d.dispatch(
        "apply_edit",
        {"path": "g.py", "edits": [{"kind": "overwrite", "old_string": "", "new_string": "y\n"}]},
    )
    assert (tmp_path / "g.py").read_text(encoding="utf-8") == "y\n"


@pytest.mark.parametrize("kind", ["create", "overwrite"])
def test_apply_edit_whole_file_kinds_share_the_create_contracts(tmp_path: Path, kind: str) -> None:
    """Both whole-file kinds refuse a non-empty old_string and refuse to be
    combined with other edits (the file stays untouched either way)."""
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("keep me\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="old_string"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": kind, "old_string": "junk", "new_string": "x"}]},
        )
    with pytest.raises(ToolError, match=kind):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [
                    {"kind": "replace", "old_string": "keep me", "new_string": "edited"},
                    {"kind": kind, "old_string": "", "new_string": "WHOLE\n"},
                ],
            },
        )
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "keep me\n"


def test_apply_patch_ok(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nb\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "path": "f.py",
            "patch": ("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"),
        },
    ).to_wire()
    assert out["path"] == "f.py"
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "a\nB\nc\n"


def test_apply_patch_create_new_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_patch",
        {
            "path": "new.py",
            "patch": "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+x = 1\n",
        },
    )
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_patch_context_mismatch_raises_tool_error(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\nWRONG\nc\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Context mismatch"):
        d.dispatch(
            "apply_patch",
            {
                "path": "f.py",
                "patch": ("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"),
            },
        )


def test_apply_patch_path_header_must_match_arg(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("a\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="disagrees"):
        d.dispatch(
            "apply_patch",
            {
                "path": "f.py",
                "patch": "--- a/g.py\n+++ b/g.py\n@@ -1 +1 @@\n-a\n+A\n",
            },
        )


def test_apply_patch_absolute_path_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch(
            "apply_patch",
            {
                "path": "/etc/passwd",
                "patch": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+A\n",
            },
        )


def test_apply_edit_preview_does_not_write(tmp_path: Path) -> None:
    """preview=true returns a diff but leaves disk untouched."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / "f.py").write_text("x = 1\n", encoding="utf-8")
    res = d.dispatch(
        "apply_edit",
        {
            "path": "f.py",
            "edits": [{"kind": "replace", "old_string": "x = 1", "new_string": "x = 99"}],
            "preview": True,
        },
    ).to_wire()
    # File on disk is unchanged.
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    # Preview payload has expected shape.
    assert res["preview"] is True
    assert res["path"] == "f.py"
    assert res["hunks"] == 1
    assert "-x = 1" in res["diff"]
    assert "+x = 99" in res["diff"]
    assert res["bytes_before"] == len("x = 1\n")
    assert res["bytes_after"] == len("x = 99\n")
    assert res["would_apply"] == ["replace"]
    assert res["truncated"] is False


def test_apply_edit_preview_for_new_file_shows_dev_null(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    res = d.dispatch(
        "apply_edit",
        {
            "path": "new.py",
            "edits": [{"kind": "create", "old_string": "", "new_string": "hello\n"}],
            "preview": True,
        },
    ).to_wire()
    assert not (tmp_path / "new.py").exists()
    assert "/dev/null" in res["diff"]
    assert res["hunks"] == 1


def test_apply_patch_preview_does_not_write(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    (tmp_path / "f.py").write_text("a\n", encoding="utf-8")
    patch = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+A\n"
    res = d.dispatch(
        "apply_patch",
        {"path": "f.py", "patch": patch, "preview": True},
    ).to_wire()
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "a\n"
    assert res["preview"] is True
    assert res["hunks"] == 1
    assert "+A" in res["diff"]


def test_apply_edit_preview_truncates_giant_diff(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    big_old = "line\n" * 5000
    big_new = "LINE\n" * 5000
    (tmp_path / "big.txt").write_text(big_old, encoding="utf-8")
    res = d.dispatch(
        "apply_edit",
        {
            "path": "big.txt",
            "edits": [{"kind": "replace", "old_string": big_old, "new_string": big_new}],
            "preview": True,
        },
    ).to_wire()
    assert res["truncated"] is True
    assert "<truncated" in res["diff"]
    # Disk untouched.
    assert (tmp_path / "big.txt").read_text(encoding="utf-8") == big_old


def test_run_command_disabled_when_no(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="not available"):
        d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert "run_command" not in d.available_tool_names()


def test_run_command_denial_is_typed_and_names_the_knob(tmp_path: Path) -> None:
    # The gate can't tell a human "no" from the ask-policy auto-deny of an
    # unattended run: the message blames neither ("denied by user" was a lie in
    # a machine subprocess) and names the config knob. ToolDenied (not a bare
    # ToolError) so the loop's sandbox-reachability heuristic can skip it: the
    # command never executed, it did not "fail in the jail".
    from agent6.tools.errors import ToolDenied

    cfg = _config_with_run_commands(tmp_path, "ask")

    def _no(_request: ApprovalRequest, /) -> ApprovalAnswer:
        return ApprovalAnswer(False, "stdin")

    d = ToolDispatcher(root=tmp_path, config=cfg, prompts=OperatorPrompts(approver=_no))
    with pytest.raises(ToolDenied, match=r"not approved \(sandbox.run_commands='ask'\)"):
        d.dispatch("run_command", {"argv": ["echo", "hi"]})


def test_list_dir(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "y.txt").write_text("y", encoding="utf-8")
    (tmp_path / ".hidden").write_text("h", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("list_dir", {"path": "."}).to_wire()
    assert "x/" in out["entries"]
    assert "y.txt" in out["entries"]
    assert ".hidden" in out["entries"]  # hidden entries are included (per the description)


def test_parse_metric_score_optional_group_is_no_score() -> None:
    """A pattern whose numeric capture group did not participate in the match
    (an alternation/optional group) yields group(1) == None; float(None) raises
    TypeError, which must be caught as "no score this turn", not propagate."""
    from agent6.tools._result_format import parse_metric_score

    # Group 1 is in the first alternative; the matched text hits the second, so
    # group(1) is None. Pre-fix this raised TypeError instead of returning None.
    assert parse_metric_score("build done", "", pattern=r"score: (\d+)|done") is None
    # A genuinely matched numeric group still parses.
    assert parse_metric_score("score: 42", "", pattern=r"score: (\d+)|done") == 42.0


def test_passthrough_env_is_fixed_allowlist() -> None:
    """Regression: dispatch must never forward LD_*/PYTHON*/DYLD_* to the jail.

    The Rust launcher does `env_clear()` before applying `policy.env`, so this
    is defense-in-depth — but if someone ever widens the allowlist without
    auditing, this test fails loudly.
    """
    from agent6.tools import _result_format

    passthrough_keys: tuple[str, ...] = _result_format.PASSTHROUGH_ENV_KEYS

    forbidden_prefixes = ("LD_", "DYLD_", "PYTHON")
    for key in passthrough_keys:
        assert not any(key.startswith(p) for p in forbidden_prefixes), (
            f"dangerous env key in allowlist: {key}"
        )
    # And at runtime, even if such vars are set in the parent, they must not
    # appear in the dict the dispatcher builds for the jail policy.
    import os

    saved = {k: os.environ.get(k) for k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH")}
    try:
        os.environ["LD_PRELOAD"] = "/tmp/evil.so"
        os.environ["LD_LIBRARY_PATH"] = "/tmp/evil"
        os.environ["PYTHONPATH"] = "/tmp/evil"
        env = _result_format.passthrough_env()
        assert "LD_PRELOAD" not in env
        assert "LD_LIBRARY_PATH" not in env
        assert "PYTHONPATH" not in env
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_jail_env_disables_python_bytecode(tmp_path: Path) -> None:
    from agent6.types import CommandResult, JailPolicy

    cfg = _config_with_run_commands(tmp_path, "yes")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    captured: dict[str, str] = {}

    def fake_run(policy: JailPolicy, **_kw: object) -> CommandResult:
        captured.update(dict(policy.env))
        return CommandResult(
            argv=("true",),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )

    with mock.patch("agent6.tools.dispatch.run_in_jail", side_effect=fake_run):
        d.dispatch("run_verify_command", {})

    assert captured["PYTHONDONTWRITEBYTECODE"] == "1"


def test_outline_returns_symbols(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\nclass Bar:\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("outline", {"path": "a.py"}).to_wire()
    names = {(s["name"], s["kind"]) for s in out["symbols"]}
    assert ("foo", "function") in names
    assert ("Bar", "class") in names
    assert out["truncated"] is False


def test_nav_tools_report_one_based_lines(tmp_path: Path) -> None:
    """outline/find_definition/find_references share the LSP twins' 1-based
    line/col convention: `class Bar` on source line 3 is line 3 on every
    surface, not tree-sitter's 0-based start_point."""
    cfg = _config(tmp_path)
    src = "def foo():\n    pass\nclass Bar:\n    pass\nfoo()\n"
    (tmp_path / "a.py").write_text(src, encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    outline = {s["name"]: s for s in d.dispatch("outline", {"path": "a.py"}).to_wire()["symbols"]}
    assert outline["Bar"]["line"] == 3
    assert outline["foo"]["line"] == 1
    assert outline["foo"]["col"] == 5  # "foo" starts at the 5th character of "def foo():"
    defs = d.dispatch("find_definition", {"symbol": "Bar"}).to_wire()["definitions"]
    assert [x["line"] for x in defs] == [3]
    refs = d.dispatch("find_references", {"symbol": "foo"}).to_wire()["references"]
    assert sorted(r["line"] for r in refs) == [1, 5]  # definition + call


def test_outline_rejects_directory(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "sub").mkdir()
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Not a file"):
        d.dispatch("outline", {"path": "sub"})


def test_outline_rejects_escape(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="Absolute"):
        d.dispatch("outline", {"path": "/etc/hosts"})


def test_find_definition_returns_relative_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def target():\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("find_definition", {"symbol": "target"}).to_wire()
    assert len(out["definitions"]) == 1
    assert out["definitions"][0]["path"] == "a.py"
    assert out["definitions"][0]["kind"] == "function"


def test_find_references_returns_relative_paths(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\nfoo()\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("find_references", {"symbol": "foo"}).to_wire()
    # Definition + call
    assert len(out["references"]) == 2
    assert all(r["path"] == "a.py" for r in out["references"])


def test_apply_edit_invalidates_index(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    # Prime the index
    assert d.dispatch("find_definition", {"symbol": "foo"}).to_wire()["definitions"]
    assert d.dispatch("find_definition", {"symbol": "bar"}).to_wire()["definitions"] == []
    # Edit the file via the tool layer.
    d.dispatch(
        "apply_edit",
        {
            "path": "a.py",
            "edits": [
                {
                    "kind": "replace",
                    "old_string": "def foo():\n    pass\n",
                    "new_string": "def bar():\n    pass\n",
                }
            ],
        },
    )
    assert d.dispatch("find_definition", {"symbol": "bar"}).to_wire()["definitions"]
    assert d.dispatch("find_definition", {"symbol": "foo"}).to_wire()["definitions"] == []


def test_new_index_tools_listed_in_available(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = set(d.available_tool_names())
    assert {"outline", "find_definition", "find_references"} <= names


def test_run_metric_command_no_config(tmp_path: Path) -> None:
    cfg = _config_with_run_commands(tmp_path, "yes")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"no \[workflow.metric\]"):
        d.dispatch("run_metric_command", {})
    # Not in the LLM-visible tool surface either.
    assert "run_metric_command" not in d.available_tool_names()


def test_run_metric_command_is_withheld_when_commands_are(tmp_path: Path) -> None:
    """It runs the operator's argv in the same jail, as often as the model
    asks, so `run_commands = "no"` withholds it with the other command tools:
    it was exposed and dispatched under "no", and prompted under "ask" only
    because the harness calls it itself."""
    body = _VALID_TOML + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"CYCLES: 42\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config

    d = ToolDispatcher(root=tmp_path, config=load_config(p))

    assert "run_metric_command" not in d.available_tool_names()
    with pytest.raises(ToolError, match="run_commands = 'no'"):
        d.dispatch("run_metric_command", {})


def test_run_metric_command_invokes_jail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"') + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"CYCLES: 42\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    cfg = load_config(p)

    captured: dict[str, object] = {}

    def fake_run_in_jail(policy, **_kw):  # type: ignore[no-untyped-def]
        captured["argv"] = tuple(policy.argv)
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=0,
            stdout="CYCLES: 42\n",
            stderr="",
            duration_s=0.01,
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("run_metric_command", {}).to_wire()
    assert out["returncode"] == 0
    assert "CYCLES: 42" in out["stdout"]
    assert captured["argv"] == ("/usr/bin/python3", "-c", 'print("CYCLES: 42")')
    # audit: the handler now parses the pattern's first capture
    # group to a float and surfaces it as `score`.
    assert out["score"] == 42.0


def test_run_metric_command_honors_verify_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify_timeout_s bounds the metric command like the verify command
    (its documented scope); the metric path silently ran on the jail's fixed
    600s default, so a bench config's fast-failure timeout never applied."""
    body = _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"') + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/true"]\n'
        'pattern = "(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    body = body.replace("[workflow]\n", "[workflow]\nverify_timeout_s = 7.0\n")
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    cfg = load_config(p)
    assert cfg.workflow.verify_timeout_s == 7.0  # the override reached the config

    captured: dict[str, object] = {}

    def fake_run_in_jail(policy, **_kw):  # type: ignore[no-untyped-def]
        captured["timeout_s"] = policy.timeout_s
        return CommandResult(
            argv=tuple(policy.argv), returncode=0, stdout="1", stderr="", duration_s=0.01
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch("run_metric_command", {})
    assert captured["timeout_s"] == 7.0


def test_run_metric_command_score_null_on_no_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pattern compiles fine but doesn't match the output -> score is null,
    rest of the result is unchanged."""
    body = _VALID_TOML.replace('run_commands = "no"', 'run_commands = "yes"') + (
        "\n[workflow.metric]\n"
        'command = ["/usr/bin/python3", "-c", "print(\\"no number here\\")"]\n'
        'pattern = "CYCLES:\\\\s*(\\\\d+)"\n'
        'goal = "minimize"\n'
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    cfg = load_config(p)

    def fake_run_in_jail(policy, **_kw):  # type: ignore[no-untyped-def]
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=0,
            stdout="no number here\n",
            stderr="",
            duration_s=0.01,
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("run_metric_command", {}).to_wire()
    assert out["score"] is None
    assert out["returncode"] == 0
    assert "no number here" in out["stdout"]


def test_disable_apply_edit_env_hides_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # AGENT6_DISABLE_APPLY_EDIT=1 removes apply_edit
    # from the surface advertised to the LLM and refuses to dispatch
    # any straggler calls. apply_patch stays available — it's the
    # whole point of the experiment.
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = d.available_tool_names()
    assert "apply_edit" not in names
    assert "apply_patch" in names


def test_disable_apply_edit_env_blocks_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT6_DISABLE_APPLY_EDIT", "1")
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="AGENT6_DISABLE_APPLY_EDIT"):
        d.dispatch(
            "apply_edit",
            {
                "path": "f.py",
                "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}],
            },
        )


def test_disable_apply_edit_unset_leaves_tool_available(tmp_path: Path) -> None:
    # Default behaviour: env var unset, both tools available.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    names = d.available_tool_names()
    assert "apply_edit" in names
    assert "apply_patch" in names


def test_dispatcher_refuses_mutations_in_plan_mode(tmp_path: Path) -> None:
    # Defense-in-depth: even if a mutation tool reaches dispatch() in plan mode
    # (the LLM's tool list already omits them), the dispatcher must refuse.
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg, mode="plan")
    with pytest.raises(ToolError, match="plan mode"):
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"kind": "create", "old_string": "", "new_string": "x\n"}]},
        )
    with pytest.raises(ToolError, match="plan mode"):
        d.dispatch("apply_patch", {"patch": "--- a\n+++ b\n"})


def test_machine_mode_blocks_edits_and_commands(tmp_path: Path) -> None:
    # a read-only machine agent state: the dispatcher refuses edits AND
    # run_command/run_verify (unlike ask, which allows run_command).
    cfg = _config_with_run_commands(tmp_path, "yes")
    d = ToolDispatcher(root=tmp_path, config=cfg, mode="machine")
    with pytest.raises(ToolError, match="machine mode"):
        d.dispatch("run_command", {"argv": ["ls"]})
    with pytest.raises(ToolError, match="machine mode"):
        d.dispatch("run_verify_command", {})
    with pytest.raises(ToolError, match="machine mode"):
        d.dispatch("apply_patch", {"patch": "--- a\n+++ b\n"})


def test_agent6_docs_tool_lists_and_reads(tmp_path: Path) -> None:
    # agent6_docs reads agent6's own bundled docs (for "how do I use agent6").
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    listing = d.dispatch("agent6_docs", {}).to_wire()
    assert "CONFIG" in listing["available"]
    assert "README" in listing["available"]
    doc = d.dispatch("agent6_docs", {"name": "CONFIG"}).to_wire()
    assert "content" in doc and len(doc["content"]) > 100
    with pytest.raises(ToolError, match="unknown agent6 doc"):
        d.dispatch("agent6_docs", {"name": "NOPE"})


# --- small-model edit ergonomics: kind default + closest-match diagnostics ---


def test_apply_edit_kind_defaults_to_replace(tmp_path: Path) -> None:
    """Small models routinely omit the `kind` discriminator. A bare
    {old_string, new_string} edit must apply as a replace, not 400."""
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_edit",
        {"path": "f.py", "edits": [{"old_string": "x = 1", "new_string": "x = 9"}]},
    ).to_wire()
    assert out["applied"] == ["replace"]
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x = 9\ny = 2\n"


def test_apply_edit_create_still_explicit(tmp_path: Path) -> None:
    """`create` is unaffected by the replace default and still needs an empty
    old_string."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch(
        "apply_edit",
        {"path": "new.py", "edits": [{"kind": "create", "new_string": "print(1)\n"}]},
    )
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)\n"


def test_apply_edit_mismatch_hands_back_exact_region(tmp_path: Path) -> None:
    """A whitespace-only mismatch returns the verbatim on-disk text and tells
    the model to retry without re-reading."""
    cfg = _config(tmp_path)
    body = (
        "class C:\n"
        "    def run(self):\n"
        "        for t in xs:\n"
        "            if t == 'dup':\n"
        "                x = self.pop()\n"
        "                self.push(x)\n"
    )
    (tmp_path / "interp.py").write_text(body, encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    # old_string with wrong (too-shallow) indentation
    bad = "        x = self.pop()\n        self.push(x)"
    with pytest.raises(ToolError) as exc:
        d.dispatch(
            "apply_edit", {"path": "interp.py", "edits": [{"old_string": bad, "new_string": "y"}]}
        )
    msg = str(exc.value)
    assert "do NOT call read_file" in msg
    assert "whitespace/indentation" in msg
    # the verbatim correct text is present so the model can copy it
    assert "                x = self.pop()\n                self.push(x)" in msg


def test_apply_edit_mismatch_unrelated_falls_back_to_shape(tmp_path: Path) -> None:
    """An old_string with no similar region gets file shape (no copyable body
    to plagiarise) and is told to re-read."""
    cfg = _config(tmp_path)
    (tmp_path / "f.py").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError) as exc:
        d.dispatch(
            "apply_edit",
            {"path": "f.py", "edits": [{"old_string": "zzz\nqqq\nwww\nvvv", "new_string": "y"}]},
        )
    msg = str(exc.value)
    assert "File shape" in msg
    assert "Re-read" in msg


# --- OpenAI V4A "*** Begin Patch" format (GPT / gpt-oss family) ---------------


def test_apply_patch_v4a_update_without_path_arg(tmp_path: Path) -> None:
    """GPT-family models emit the V4A format and omit `path` (it is in the
    patch). agent6 must parse it, derive the path, and apply the hunk."""
    cfg = _config(tmp_path)
    (tmp_path / "m.py").write_text("def f():\n    x = 1\n    return x\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    patch = (
        "*** Begin Patch\n"
        "*** Update File: m.py\n"
        "@@ def f():\n"
        "     x = 1\n"
        "-    return x\n"
        "+    return x + 1\n"
        "*** End Patch"
    )
    out = d.dispatch("apply_patch", {"patch": patch}).to_wire()
    assert out["path"] == "m.py"
    assert (tmp_path / "m.py").read_text(
        encoding="utf-8"
    ) == "def f():\n    x = 1\n    return x + 1\n"


def test_apply_patch_v4a_add_file(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    patch = "*** Begin Patch\n*** Add File: new.py\n+print(1)\n+print(2)\n*** End Patch"
    d.dispatch("apply_patch", {"patch": patch})
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "print(1)\nprint(2)\n"


def test_apply_patch_v4a_path_into_git_still_refused(tmp_path: Path) -> None:
    """Deriving the path from the patch never bypasses the protected-path guard:
    a V4A patch targeting .git is refused like any other write."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    patch = "*** Begin Patch\n*** Add File: .git/hooks/pre-commit\n+#!/bin/sh\n+id\n*** End Patch"
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch("apply_patch", {"patch": patch})


def test_apply_patch_unified_still_works_and_path_optional(tmp_path: Path) -> None:
    """The unified-diff path is unchanged and also accepts an omitted `path`
    (derived from the `+++` header)."""
    cfg = _config(tmp_path)
    (tmp_path / "x.py").write_text("a\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch", {"patch": "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+A\n"}
    ).to_wire()
    assert out["path"] == "x.py"
    assert (tmp_path / "x.py").read_text(encoding="utf-8") == "A\n"


def test_rejected_tool_emits_call_and_result_pair(tmp_path: Path) -> None:
    """A guard-rejected tool (unknown name / disabled / wrong-mode) still emits a
    tool.call + tool.result(ok=false) pair with a trusted, deterministic reason --
    so a reader never sees a loop.tool.call with no matching result, and the
    ok=false signal is dispatcher-owned (a prompt injection can't fake success)."""
    import json

    from agent6.events import EventSink

    cfg = _config(tmp_path)  # run_commands = "no"
    logs = tmp_path / "logs.jsonl"
    d = ToolDispatcher(root=tmp_path, config=cfg, events=EventSink(logs))

    with pytest.raises(ToolError):  # run_command disabled by config -> guard reject
        d.dispatch("run_command", {"argv": ["echo", "hi"]})
    with pytest.raises(ToolError):  # unknown tool name -> guard reject
        d.dispatch("totally_unknown_tool", {})

    events = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    calls = [e for e in events if e["type"] == "tool.call"]
    results = [e for e in events if e["type"] == "tool.result"]
    assert [e["name"] for e in calls] == ["run_command", "totally_unknown_tool"]
    assert [(e["name"], e["ok"]) for e in results] == [
        ("run_command", False),
        ("totally_unknown_tool", False),
    ]
    # Reasons come from the dispatcher's own guard messages, not model content.
    assert "not available" in results[0]["summary"]
    assert "Unknown tool" in results[1]["summary"]


def test_run_command_result_carries_output_tails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execution tools' tool.result events carry capped stdout/stderr tails (like
    verify.end), so logs.jsonl shows command output -- not just an exit code --
    while non-execution tools stay summary-only (full output is in transcripts)."""
    import json

    from agent6.events import EventSink

    cfg = _config_with_run_commands(tmp_path, "yes")  # skip the approval prompt
    logs = tmp_path / "logs.jsonl"
    d = ToolDispatcher(root=tmp_path, config=cfg, events=EventSink(logs))

    def _fake_run_argv(self: object, argv: object, **kw: object) -> object:
        from agent6.tools.results import ExecResult

        return ExecResult(
            returncode=1,
            stdout="OUT-X" * 500,
            stderr="ERR-Y" * 500,
            duration_s=0.1,
            exec_failed=False,
        )

    monkeypatch.setattr(ToolDispatcher, "_run_argv_in_jail", _fake_run_argv)
    d.dispatch("run_command", {"argv": ["echo", "hi"]})
    (tmp_path / "f.txt").write_text("hi", encoding="utf-8")
    d.dispatch("read_file", {"path": "f.txt"})

    results = [
        e
        for e in (json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines())
        if e["type"] == "tool.result"
    ]
    run = next(e for e in results if e["name"] == "run_command")
    assert run["ok"] is True
    assert run["stdout_tail"].startswith("OUT-X") and len(run["stdout_tail"]) == 2000
    assert run["stderr_tail"].startswith("ERR-Y") and len(run["stderr_tail"]) == 2000
    read = next(e for e in results if e["name"] == "read_file")
    assert "stdout_tail" not in read and "stderr_tail" not in read  # non-exec
    # read_file carries its own excerpt instead: a head preview + the true
    # line count, so a transcript can show what was read.
    assert read["head_tail"] == "hi"
    assert read["lines_total"] == 1


def test_run_command_passes_extra_read_paths_to_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sandbox.extra_read_paths must reach the JailPolicy as extra_ro_paths, so a
    # project whose toolchain/interpreter lives outside the repo (e.g. a conda
    # env at /opt) is usable under hardened/strict.
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    body = _VALID_TOML.replace(
        'run_commands = "no"',
        'run_commands = "yes"\nextra_read_paths = ["/opt/miniconda3"]',
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    cfg = load_config(p)
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run_in_jail(policy, **_kw):  # type: ignore[no-untyped-def]
        captured["ro"] = tuple(str(x) for x in policy.extra_ro_paths)
        return CommandResult(
            argv=tuple(policy.argv), returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert "/opt/miniconda3" in captured["ro"]


def test_run_command_passes_extra_write_paths_to_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # sandbox.extra_write_paths must reach the JailPolicy as extra_rw_paths (a
    # writable bind mount at the real location, so write implies read).
    from agent6.config import load_config
    from agent6.sandbox.jail import CommandResult

    body = _VALID_TOML.replace(
        'run_commands = "no"',
        'run_commands = "yes"\nextra_write_paths = ["/var/cache/shared"]',
    )
    p = tmp_path / "agent6.toml"
    p.write_text(body, encoding="utf-8")
    cfg = load_config(p)
    captured: dict[str, tuple[str, ...]] = {}

    def fake_run_in_jail(policy, **_kw):  # type: ignore[no-untyped-def]
        captured["rw"] = tuple(str(x) for x in policy.extra_rw_paths)
        return CommandResult(
            argv=tuple(policy.argv), returncode=0, stdout="", stderr="", duration_s=0.0
        )

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", fake_run_in_jail)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    d.dispatch("run_command", {"argv": ["echo", "hi"]})
    assert "/var/cache/shared" in captured["rw"]


# --- stringified-JSON argument coercion --------------------------------------
# A weak model sends a structured argument as a JSON string (sometimes with
# trailing junk). The dispatcher parses the head and retries once instead of
# burning a round-trip on a validation error.


def test_stringified_edits_array_is_coerced(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.txt").write_text("old text\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    edits_str = '[{"old_string": "old text", "new_string": "new text"}]\n</invoke>'
    d.dispatch("apply_edit", {"path": "a.txt", "edits": edits_str})
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "new text\n"


def test_stringified_coercion_surfaces_original_error_when_wrong(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    # Parses as JSON but has the wrong inner shape: re-validation fails and the
    # original tuple_type error (not the retry's) reaches the caller.
    with pytest.raises(ToolError, match=r"tuple_type|valid tuple|list"):
        d.dispatch("apply_edit", {"path": "a.txt", "edits": '[{"nope": 1}]'})


def test_non_json_string_still_fails_validation(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError):
        d.dispatch("apply_edit", {"path": "a.txt", "edits": "not json at all"})


def test_under_system_root_classifies_bin_dirs() -> None:
    from agent6.sandbox.jail import _under_system_root  # pyright: ignore[reportPrivateUsage]

    assert _under_system_root(Path("/usr/local/bin"))  # under a mounted system root
    assert _under_system_root(Path("/usr/bin"))
    assert not _under_system_root(Path("/opt/pipx/venvs/uv/bin"))  # pipx target
    assert not _under_system_root(Path("/home/x/.local/bin"))


def test_operator_tool_paths_extends_path_and_mounts_are_nonsystem() -> None:
    from agent6.sandbox.jail import (
        _under_system_root,  # pyright: ignore[reportPrivateUsage]
        operator_tool_paths,
    )

    path, mounts = operator_tool_paths()
    # PATH always starts with the jail baseline, then any standard bin dirs.
    assert path.startswith("/usr/bin:/bin")
    # Mounts are only dirs OUTSIDE the system roots (those are already mounted via
    # /usr); a system dir here would be a redundant/failing re-bind.
    for m in mounts:
        assert not _under_system_root(m), m
        assert m.is_dir()


def test_operator_tool_paths_mounts_uv_managed_pythons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A repo venv made by uv can symlink python to a uv-managed CPython under
    # XDG data; without the RO mount an in-jail `uv run` sees a "non-existent
    # interpreter" and deletes + recreates the operator's .venv.
    from agent6.sandbox.jail import operator_tool_paths

    pythons = tmp_path / "uv" / "python"
    pythons.mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    path, mounts = operator_tool_paths()
    assert pythons in mounts
    assert str(pythons) not in path  # mount-only: interpreters are not PATH dirs

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "elsewhere"))
    _, mounts = operator_tool_paths()
    assert pythons not in mounts  # absent dir -> no mount


def test_ask_user_accepts_flat_single_question(tmp_path: Path) -> None:
    # A model that sends a lone question flat (not wrapped in `questions`) still works.
    cfg = _config(tmp_path)

    def questioner(request: QuestionRequest, /) -> QuestionAnswer:
        answers = tuple(q.options[0] if q.options else "typed" for q in request.questions)
        return QuestionAnswer(answers, "stdin")

    d = ToolDispatcher(root=tmp_path, config=cfg, prompts=OperatorPrompts(questioner=questioner))
    out = d.dispatch(
        "ask_user", {"question": "Which theme?", "options": ["dark", "light"]}
    ).to_wire()
    assert out == {"answers": ["dark"]}


@pytest.mark.parametrize(("isolation", "protected"), [("strict", True), ("hardened", False)])
def test_git_reaches_the_jail_as_a_protect_path_only_under_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolation: IsolationLevel, protected: bool
) -> None:
    """`protect_git` is a read-only bind, so it needs a mount namespace.

    The threat is real on hardened too -- a jailed command can plant a
    `filter.<n>.clean` in .git/config with a matching .gitattributes, and
    agent6's own host-side auto-commit (`git add -A`, run OUTSIDE the jail)
    then executes it. But Landlock cannot express the protection: a grant on a
    directory is recursive and stacked rulesets only intersect, so denying
    `.git` means not granting the workspace root either -- which denied every
    `touch`/`mkdir` at the root. Hardened warns instead, and an explicitly-set
    `protect_git = true` refuses to run there."""
    (tmp_path / ".git").mkdir()
    captured: list[object] = []

    from agent6.types import CommandResult

    def _capture(policy: object, **_kw: object) -> CommandResult:
        captured.append(policy)
        return CommandResult(argv=("true",), returncode=0, stdout="", stderr="", duration_s=0.0)

    monkeypatch.setattr("agent6.tools.dispatch.run_in_jail", _capture)
    cfg = _config_with_run_commands(tmp_path, "yes")
    d = ToolDispatcher(root=tmp_path, config=cfg, isolation=isolation)
    d.dispatch("run_command", {"argv": ["true"]})

    paths = captured[0].extra_protect_paths  # pyright: ignore[reportAttributeAccessIssue]
    assert ((tmp_path / ".git").resolve() in paths) is protected


def test_every_jail_tool_answers_to_run_commands(tmp_path: Path) -> None:
    """run_verify_command runs model-influenced argv in the same jail with the
    same reach as run_command -- its argv is INFERRED from a file the model can
    edit whenever it is not configured -- so it answers to the same knob rather
    than running while `run_commands = "no"` disables everything else."""
    gated = {"run_command", "run_verify_command", "stop_background"}
    denied = ToolDispatcher(root=tmp_path, config=_config_with_run_commands(tmp_path, "no"))
    assert gated.isdisjoint(denied.available_tool_names())
    for name in sorted(gated):
        with pytest.raises(ToolError, match="run_commands"):
            denied.dispatch(name, {"argv": ["true"], "id": "bg1"})
    allowed = ToolDispatcher(root=tmp_path, config=_config_with_run_commands(tmp_path, "yes"))
    assert gated <= set(allowed.available_tool_names())


def test_ask_prompts_before_the_verify_gate_runs(tmp_path: Path) -> None:
    """Under `ask` the operator approves the verify command like any other, and
    a refusal denies the call instead of running it."""
    from agent6.tools.errors import ToolDenied

    asked: list[str] = []

    def refuse(request: ApprovalRequest, /) -> ApprovalAnswer:
        asked.append(request.prompt)
        return ApprovalAnswer(False, "stdin")

    d = ToolDispatcher(
        root=tmp_path,
        config=_config_with_run_commands(tmp_path, "ask"),
        prompts=OperatorPrompts(approver=refuse),
    )
    with pytest.raises(ToolDenied):
        d.dispatch("run_verify_command", {})
    assert asked and asked[0].startswith("Allow run_verify_command: true")


def test_operator_tool_paths_never_mounts_agent6s_own_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool symlink must not drag agent6's private dirs into the jail.

    `operator_tool_paths` mounts `real.parent` for every symlink in a bin dir.
    A symlink resolving into the config dir mounted `secrets.toml` (provider API
    keys) read-only into the jail; one into the state dir mounted notes,
    memories and transcripts -- against the invariant docs/security.md states.
    Not model-reachable (a jailed `ln -s` into a bin dir is refused), but the
    invariant should hold by construction, not by luck of directory layout.
    """
    from agent6.sandbox.jail import operator_tool_paths

    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    # The authoritative overrides (conftest sets these too, so XDG_* is ignored).
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(home / ".config" / "agent6"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(home / ".local" / "state" / "agent6"))

    private = {
        "cfg": home / ".config" / "agent6",
        "state": home / ".local" / "state" / "agent6" / "repo-1",
        "ordinary": home / "tools" / "bin",  # the control: this MUST still mount
    }
    for name, d in private.items():
        d.mkdir(parents=True)
        exe = d / f"{name}-tool"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        exe.chmod(0o755)
        (bin_dir / f"link-{name}").symlink_to(exe)

    _, mounts = operator_tool_paths()

    assert private["cfg"] not in mounts, "a symlink mounted the dir holding secrets.toml"
    assert private["state"] not in mounts, "a symlink mounted the per-repo state dir"
    # The converse, so the fix cannot degrade into "mount nothing": an ordinary
    # tool dir a symlink resolves into is still mounted.
    assert private["ordinary"] in mounts, "a legitimate tool mount was dropped"


def test_a_tool_mount_never_contains_a_private_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test above pins mounts INSIDE a private dir; containment fails the
    other way round too. A symlink out to `<dir>/x.sh` mounts `<dir>` whole --
    and a `<dir>` holding the config dir grants secrets.toml from above."""
    from agent6.sandbox.jail import operator_tool_paths

    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    holder = tmp_path / "xdg"  # not home and not an ancestor of it
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(holder / "agent6-config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(holder / "agent6-state"))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(holder / "agent6-data"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(holder / "agent6-cache"))
    (holder / "agent6-config").mkdir(parents=True)

    target = holder / "x.sh"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    (bin_dir / "x").symlink_to(target)

    _, mounts = operator_tool_paths()
    assert holder not in mounts, "a mount containing the config dir grants secrets.toml"


def test_home_and_its_ancestors_are_never_tool_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with every agent6 dir elsewhere, $HOME holds ~/.ssh and every
    credential the operator owns. A plain `~/.local/bin/x -> ~/x.sh` makes
    `real.parent` the whole home dir; an ancestor contains home in turn."""
    from agent6.sandbox.jail import operator_tool_paths

    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(elsewhere / "config"))
    monkeypatch.setenv("AGENT6_STATE_HOME", str(elsewhere / "state"))
    monkeypatch.setenv("AGENT6_DATA_HOME", str(elsewhere / "data"))
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(elsewhere / "cache"))

    for name, target_dir in {"home-tool": home, "ancestor-tool": tmp_path}.items():
        target = target_dir / f"{name}.sh"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
        (bin_dir / name).symlink_to(target)

    _, mounts = operator_tool_paths()
    assert home not in mounts, "a symlink to ~/x.sh mounted the whole home dir"
    assert tmp_path not in mounts, "a symlink target above home mounted homes ancestor"


def test_edit_tools_name_a_directory_like_their_siblings_do(tmp_path: Path) -> None:
    """`read_file` on a directory says "Not a file: x". apply_edit/apply_patch
    leaked the raw errno instead ("[Errno 21] Is a directory: /abs/path"), which
    also puts an absolute host path in the model's transcript."""
    cfg = _config(tmp_path)
    (tmp_path / "adir").mkdir()
    d = ToolDispatcher(root=tmp_path, config=cfg)

    for tool, args in (
        ("apply_edit", {"path": "adir", "edits": [{"old_string": "a", "new_string": "b"}]}),
        ("apply_patch", {"path": "adir", "patch": "--- a/adir\n+++ b/adir\n"}),
    ):
        with pytest.raises(ToolError) as exc:
            d.dispatch(tool, args)
        assert "Not a file: adir" in str(exc.value), f"{tool}: {exc.value}"
        assert "Errno" not in str(exc.value), f"{tool} leaked an errno: {exc.value}"


def test_read_file_refuses_a_binary_file_as_its_description_promises(tmp_path: Path) -> None:
    """The tool description says read_file "fails when the file is binary", but
    it only caught UnicodeDecodeError -- so a file with NUL bytes that happens
    to decode as UTF-8 was returned verbatim into the transcript. A NUL byte is
    the definition of binary; make the promise true rather than soften it."""
    cfg = _config(tmp_path)
    (tmp_path / "b.bin").write_bytes(b"text\x00\x01more\n")
    d = ToolDispatcher(root=tmp_path, config=cfg)

    with pytest.raises(ToolError, match="binary"):
        d.dispatch("read_file", {"path": "b.bin"})

    # The converse: ordinary text still reads.
    (tmp_path / "ok.txt").write_text("hello\n", encoding="utf-8")
    assert d.dispatch("read_file", {"path": "ok.txt"}).to_wire()["content"] == "hello\n"


def test_apply_patch_multi_file_v4a(tmp_path: Path) -> None:
    """A multi-file V4A patch (what GPT-family models emit natively) applies
    every file all-or-nothing. One-file-per-call was a pre-1.0 placeholder;
    SWE-bench transcripts showed models looping on the rejection."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("p\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "patch": (
                "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                "*** Update File: b.py\n@@\n-p\n+q\n*** End Patch"
            ),
        },
    ).to_wire()
    assert out["files"] == [
        {"path": "a.py", "bytes_written": 2},
        {"path": "b.py", "bytes_written": 2},
    ]
    assert out["path"] == "a.py" and out["bytes_written"] == 4
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "y\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "q\n"


def test_apply_patch_multi_file_unified_diff_git(tmp_path: Path) -> None:
    """git-style multi-file unified diffs split at `diff --git` boundaries."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "patch": (
                "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-x\n+y\n"
                "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n"
                "@@ -0,0 +1 @@\n+fresh\n"
            ),
        },
    ).to_wire()
    assert [f["path"] for f in out["files"]] == ["a.py", "new.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "y\n"
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "fresh\n"


def test_apply_patch_multi_file_is_all_or_nothing(tmp_path: Path) -> None:
    """A context miss in the SECOND file leaves the first unwritten."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("DIFFERENT\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"b\.py"):
        d.dispatch(
            "apply_patch",
            {
                "patch": (
                    "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                    "*** Update File: b.py\n@@\n-p\n+q\n*** End Patch"
                ),
            },
        )
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x\n"


def test_apply_patch_multi_file_protected_second_file_writes_nothing(tmp_path: Path) -> None:
    """The protected-path guard runs per file BEFORE any write: a .git target
    anywhere in a multi-file patch refuses the whole call."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_patch",
            {
                "patch": (
                    "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                    "*** Add File: .git/hooks/pre-commit\n+#!/bin/sh\n*** End Patch"
                ),
            },
        )
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x\n"


def test_apply_patch_multi_file_path_arg_rejected(tmp_path: Path) -> None:
    """`path` cannot name the target of a multi-file patch."""
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="ambiguous"):
        d.dispatch(
            "apply_patch",
            {
                "path": "a.py",
                "patch": (
                    "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                    "*** Update File: b.py\n@@\n-p\n+q\n*** End Patch"
                ),
            },
        )


def test_apply_patch_deletes_a_file(tmp_path: Path) -> None:
    """Deletion via patch (unified `+++ /dev/null` asserts the full content;
    V4A `*** Delete File:` deletes by name). One-file-per-op with a
    run_command rm fallback was the pre-1.0 placeholder."""
    cfg = _config(tmp_path)
    (tmp_path / "gone.py").write_text("a\nb\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {"patch": "--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n"},
    ).to_wire()
    assert out == {"path": "gone.py", "bytes_written": 0, "deleted": ["gone.py"]}
    assert not (tmp_path / "gone.py").exists()


def test_apply_patch_v4a_delete_in_multi_file(tmp_path: Path) -> None:
    """An update + a delete in one V4A patch apply all-or-nothing."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "old.py").write_text("junk\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "patch": (
                "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                "*** Delete File: old.py\n*** End Patch"
            ),
        },
    ).to_wire()
    assert out["files"] == [{"path": "a.py", "bytes_written": 2}]
    assert out["deleted"] == ["old.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "y\n"
    assert not (tmp_path / "old.py").exists()


def test_apply_patch_delete_missing_file_refused(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="no such file"):
        d.dispatch(
            "apply_patch",
            {"patch": "*** Begin Patch\n*** Delete File: ghost.py\n*** End Patch"},
        )


def test_apply_patch_delete_protected_path_refused(tmp_path: Path) -> None:
    """Deletion runs the same protected-path guard as writes."""
    cfg = _config(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match=r"\.git"):
        d.dispatch(
            "apply_patch",
            {"patch": "*** Begin Patch\n*** Delete File: .git/config\n*** End Patch"},
        )
    assert (tmp_path / ".git" / "config").exists()


def test_apply_patch_multi_file_preview_concatenates(tmp_path: Path) -> None:
    """preview=true over a multi-file patch returns every file's diff and
    writes nothing (it refused before)."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("p\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {
            "preview": True,
            "patch": (
                "*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n"
                "*** Update File: b.py\n@@\n-p\n+q\n*** End Patch"
            ),
        },
    ).to_wire()
    assert out["preview"] is True
    assert out["files"] == ["a.py", "b.py"]
    assert "a/a.py" in out["diff"] and "a/b.py" in out["diff"]
    assert out["hunks"] == 2
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "p\n"


def test_apply_patch_reports_heals_on_the_wire(tmp_path: Path) -> None:
    """A healed hunk applied, but not verbatim: the wire says so, so the
    model knows its context was off instead of trusting a silent success."""
    cfg = _config(tmp_path)
    (tmp_path / "a.py").write_text("def f():\n    a = 1\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch(
        "apply_patch",
        {"patch": ("*** Begin Patch\n*** Update File: a.py\n@@\n-a = 1\n+a = 10\n*** End Patch")},
    ).to_wire()
    assert out["healed"] == ["a.py ~indent"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f():\n    a = 10\n"


def test_an_omitted_edit_kind_follows_the_pair_it_was_sent_with(tmp_path: Path) -> None:
    """Small models send a bare {old_string, new_string}. An empty old_string
    can only mean "write this whole file", and defaulting it to `replace`
    refused the natural write-a-new-file call forever: a live `machine create`
    spent three attempts on it and stopped on the tool-error streak.

    It resolves to `create`, never `overwrite`, so a model that thinks a file
    is new still cannot clobber one that exists.
    """
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)

    d.dispatch("apply_edit", {"path": "new.py", "edits": [{"new_string": "x = 1\n"}]})
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    # ... and the same call over an existing file refuses rather than clobbering
    with pytest.raises(ToolError, match="already exists"):
        d.dispatch("apply_edit", {"path": "new.py", "edits": [{"new_string": "y = 2\n"}]})
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"

    # a non-empty old_string still means a replace
    d.dispatch(
        "apply_edit", {"path": "new.py", "edits": [{"old_string": "x = 1", "new_string": "z = 3"}]}
    )
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "z = 3\n"


def test_two_patch_sections_over_one_file_are_refused(tmp_path: Path) -> None:
    """Each section is staged against the file as it is ON DISK, so two
    sections over one file both start from the original and the last write
    wins: the earlier edit vanished while the result reported it applied, and
    double-counted its bytes."""
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    (tmp_path / "m.py").write_text("A = 1\nB = 2\nC = 3\nD = 4\nE = 5\n", encoding="utf-8")
    patch = (
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -1,3 +1,3 @@\n A = 1\n-B = 2\n+B = 22\n C = 3\n"
        "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n"
        "@@ -3,3 +3,3 @@\n C = 3\n-D = 4\n+D = 44\n E = 5\n"
    )

    with pytest.raises(ToolError, match="appears in 2 sections"):
        d.dispatch("apply_patch", {"patch": patch})

    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "A = 1\nB = 2\nC = 3\nD = 4\nE = 5\n"

    # The preview shows what the apply would do, so it refuses the same patch
    # instead of previewing a diff whose second section reads the ORIGINAL.
    with pytest.raises(ToolError, match="appears in 2 sections"):
        d.dispatch("apply_patch", {"patch": patch, "preview": True})

    # And the count names the repeated file, not the patch's section total.
    three = patch + ("diff --git a/o.py b/o.py\n--- /dev/null\n+++ b/o.py\n@@ -0,0 +1 @@\n+O = 1\n")
    with pytest.raises(ToolError, match=r"m\.py appears in 2 sections"):
        d.dispatch("apply_patch", {"patch": three})

    # By its repo path: two `m.py` under different directories are two files,
    # and a basename cannot tell the model which one it repeated.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("A = 1\nB = 2\nC = 3\n", encoding="utf-8")
    nested = (
        "diff --git a/pkg/m.py b/pkg/m.py\n--- a/pkg/m.py\n+++ b/pkg/m.py\n"
        "@@ -1,3 +1,3 @@\n A = 1\n-B = 2\n+B = 22\n C = 3\n"
    ) * 2
    with pytest.raises(ToolError, match=r"pkg/m\.py appears in 2 sections"):
        d.dispatch("apply_patch", {"patch": nested})


def test_a_patch_write_that_fails_part_way_names_what_changed(tmp_path: Path) -> None:
    """Staging is all-or-nothing; the writes were not, and a second file the
    process could not write reported a failure over a first file it had
    already rewritten, with nothing telling the model so."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root writes through a read-only mode")
    cfg = _config(tmp_path)
    (tmp_path / "one.txt").write_text("original one\n", encoding="utf-8")
    (tmp_path / "locked.txt").write_text("locked original\n", encoding="utf-8")
    (tmp_path / "locked.txt").chmod(0o444)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    patch = (
        "diff --git a/one.txt b/one.txt\n--- a/one.txt\n+++ b/one.txt\n"
        "@@ -1,1 +1,1 @@\n-original one\n+PATCHED ONE\n"
        "diff --git a/locked.txt b/locked.txt\n--- a/locked.txt\n+++ b/locked.txt\n"
        "@@ -1,1 +1,1 @@\n-locked original\n+PATCHED LOCKED\n"
    )
    with pytest.raises(ToolError, match=r"locked\.txt.*already changed: one\.txt"):
        d.dispatch("apply_patch", {"patch": patch})
    assert (tmp_path / "one.txt").read_text(encoding="utf-8") == "PATCHED ONE\n"


def test_the_edit_tools_refuse_a_file_past_the_read_cap(tmp_path: Path) -> None:
    """read_file caps at MAX_READ_CHARS; the edit tools read the same file
    whole and uncapped, so one `apply_edit` over a file a jailed command had
    made OOM-crashed the unsandboxed agent. Refused, never truncated: a
    partial read must not become a whole-file write."""
    from agent6.tools._fs_tools import MAX_READ_CHARS

    cfg = _config(tmp_path)
    (tmp_path / "big.txt").write_text("x" * (MAX_READ_CHARS + 1), encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    with pytest.raises(ToolError, match="larger than the edit tools take"):
        d.dispatch(
            "apply_edit", {"path": "big.txt", "edits": [{"old_string": "x", "new_string": "y"}]}
        )


def test_list_dir_caps_a_huge_listing_and_says_so(tmp_path: Path) -> None:
    """Every sibling result caps and marks; a listing of a vendored directory
    went whole into the context and the transcript."""
    from agent6.tools._fs_tools import LIST_DIR_CAP

    cfg = _config(tmp_path)
    big = tmp_path / "vendored"
    big.mkdir()
    for i in range(LIST_DIR_CAP + 1):
        (big / f"f{i:05d}").write_text("", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    out = d.dispatch("list_dir", {"path": "vendored"}).to_wire()
    assert len(out["entries"]) == LIST_DIR_CAP
    assert out["truncated"] is True


def test_sizes_on_the_wire_are_bytes(tmp_path: Path) -> None:
    """`size`, `bytes_written` and the "N bytes" summary counted characters:
    ten `é` read as 11 bytes where the file holds 21."""
    cfg = _config(tmp_path)
    (tmp_path / "u.txt").write_text("é" * 10 + "\n", encoding="utf-8")
    d = ToolDispatcher(root=tmp_path, config=cfg)
    read = d.dispatch("read_file", {"path": "u.txt"})
    assert read.to_wire()["size"] == 21 and "21 bytes" in read.summary()
    patch = "--- a/u.txt\n+++ b/u.txt\n@@ -1,1 +1,1 @@\n-" + "é" * 10 + "\n+" + "é" * 10 + "!\n"
    preview = d.dispatch("apply_patch", {"patch": patch, "preview": True}).to_wire()
    assert preview["bytes_after"] == 22
    written = d.dispatch("apply_patch", {"patch": patch}).to_wire()
    assert written["bytes_written"] == 22


def test_the_argument_preview_clips_at_every_depth() -> None:
    """The `tool.call` preview clipped top-level strings only, so an
    apply_edit's `edits[0].new_string` wrote a whole file into the durable
    event log per call while the sibling apply_patch clipped to 200 chars."""
    from agent6.tools._result_format import truncate_args

    out = truncate_args({"path": "a.py", "edits": [{"old_string": "x", "new_string": "y" * 5000}]})
    clipped = out["edits"][0]["new_string"]
    assert len(clipped) < 300 and clipped.endswith("… (5000 chars)")
    assert out["path"] == "a.py"
