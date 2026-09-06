# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drift pins: prose that names code must track the code.

Each pin failed (or nearly failed) in the wild before it existed: a tool
renamed out from under a prompt mention, a bench config key that did not
match the Config field, a dependency count written as a word."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from agent6.prompts import loop as prompts
from agent6.tools import schema as tool_schema

REPO = Path(__file__).resolve().parents[2]

_PROMPT_CONSTANTS = (
    prompts.SYSTEM_PROMPT_BASE,
    prompts.PLAN_SYSTEM_PROMPT_BASE,
    prompts.ASK_SYSTEM_PROMPT_BASE,
    prompts.MACHINE_SYSTEM_PROMPT_BASE,
    prompts.AGENT_SYSTEM_PROMPT_BASE,
    prompts.GIT_PROTECT_RULE,
    prompts.NO_AUTO_COMMIT_RULE,
    prompts.HARDENED_FS_RULE,
    prompts.DAG_RULES_OPTIONAL,
    prompts.DAG_RULES_DECOMPOSE,
)

# Backticked identifiers in the prompts that are deliberately NOT tool names:
# tool parameters, task statuses, file formats, code symbols.
_NON_TOOL_SPANS = {
    "acceptance",
    "agent",
    "build_system_prompt",
    "depends_on",
    "obsolete",
    "parent_id",
    "result",
    "skipped",
    "summary",
    "stale_gate",
    "title",
    "toml",
    "old_string",
    "new_string",
    "kind",
    "preview",
    "in_progress",
    "passed",
    "standing",
}


def _registered_tool_names() -> set[str]:
    names = {cls.TOOL_NAME for cls in tool_schema.ALL_TOOLS}
    for mode in ("run", "plan", "ask", "machine", "agent"):
        mt = tool_schema.mode_tools(mode)
        names |= {cls.TOOL_NAME for cls in (*mt.base, *mt.extras)}
    return names


def test_every_tool_mention_in_the_prompts_is_a_registered_tool() -> None:
    registered = _registered_tool_names()
    for text in _PROMPT_CONSTANTS:
        for span in re.findall(r"`([a-z][a-z0-9_]+)`", text):
            assert span in registered or span in _NON_TOOL_SPANS, (
                f"prompt mentions `{span}`, which is neither a registered tool"
                " nor in the non-tool allowlist: a rename left stale text, or"
                " the allowlist needs the new parameter"
            )


def test_readme_dependency_count_matches_pyproject() -> None:
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    words = {5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
    readme = (REPO / "README.md").read_text()
    m = re.search(r"(\w+) runtime dependencies", readme)
    assert m is not None, "README no longer states the dependency count"
    assert m.group(1) == words.get(len(deps), str(len(deps))), (
        f"README says '{m.group(1)} runtime dependencies' but pyproject has {len(deps)}"
    )


def test_readme_quoted_security_defaults_match_config() -> None:
    from agent6.config import Config

    cfg = Config()
    readme = (REPO / "README.md").read_text()
    for quoted, actual in (
        ('network = "auto"', cfg.sandbox.network),
        ('run_commands = "ask"', cfg.sandbox.run_commands),
        ("protect_git = true", cfg.sandbox.protect_git),
    ):
        assert quoted in readme, f"README no longer quotes {quoted!r}"
        want = quoted.split("= ")[1].strip('"')
        assert str(actual).lower() == want, (
            f"README quotes {quoted!r} but the Config default is {actual!r}"
        )


def test_bench_container_config_template_validates() -> None:
    """Render in_container.sh's config heredoc with stub values and validate it
    through Config: a key that does not match a field refuses every bench run
    (verify_timeout vs verify_timeout_s was caught by hand once)."""
    from agent6.config import Config

    script = (REPO / "bench" / "swebench" / "in_container.sh").read_text()
    m = re.search(r"cat > /root/agent6\.toml <<EOF\n(.*?)\nEOF\n", script, re.S)
    assert m is not None, "config heredoc not found in in_container.sh"
    stub = {
        "PROVIDER_BLOCK": (
            '[providers.openrouter]\napi_format = "openai"\napi_key_env = "K"\n'
            'base_url = "https://openrouter.ai/api/v1"'
        ),
        "PROVIDER": "openrouter",
        "MODEL": "moonshotai/stub",
        "MAX_USD": "1.0",
        "REVIEW_LINES": "",
        "VERIFY_TOML": 'verify_command = ["true"]',
        "PROMPT_FILE_LINE": "",
        "AGENT6_SB_STRUCTURAL_PRIORS": "",
    }

    def _render(env: dict[str, str]) -> str:
        def _sub(mo: re.Match[str]) -> str:
            name = mo.group(1) or mo.group(2)
            op, text = mo.group(3), mo.group(4)
            val = env.get(name, "")
            if op == "+":  # ${VAR:+text}: text (with \" unescaped) iff VAR set
                return text.replace('\\"', '"') if val else ""
            return val if val else (text or "")

        out = m.group(1)
        for _ in range(2):  # ${VAR:+...$VAR...} nests one level
            out = re.sub(r"\$(?:([A-Z0-9_]+)|\{([A-Z0-9_]+)(?::([-+])([^}]*))?\})", _sub, out)
        return out

    for env in (stub, {**stub, "AGENT6_SB_EFFORT": "medium"}):
        rendered = _render(env)
        data = tomllib.loads(rendered)
        data.pop("agent6", None)
        Config.model_validate(data)
        if env.get("AGENT6_SB_EFFORT"):
            assert data["models"]["worker"]["effort"] == "medium"
