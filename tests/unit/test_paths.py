# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.paths (XDG resolution, sudo/root handling)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent6 import paths


def test_global_config_dir_follows_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)  # not root: XDG is honored
    assert paths.global_config_dir() == tmp_path / "g" / "agent6"
    assert paths.global_config_path() == tmp_path / "g" / "agent6" / "config.toml"
    assert paths.secrets_path() == tmp_path / "g" / "agent6" / "secrets.toml"


def test_state_dir_and_repo_config_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(base))
    repo = tmp_path / "myrepo"
    repo.mkdir()
    rid = paths.repo_id(repo)
    # The id names the workspace's whole path, so a state dir says where it came
    # from; the trailing tag is what keeps two workspaces apart.
    assert "myrepo" in rid
    assert rid.rsplit("-", 1)[1].strip("0123456789abcdef") == ""
    assert "/" not in rid
    assert paths.repo_id(repo) == rid  # deterministic
    assert paths.state_dir(repo) == base / "agent6" / rid
    assert paths.repo_config_path(repo) == base / "agent6" / rid / "config.toml"


def test_state_tree_dirs_are_created_private_0700(tmp_path: Path) -> None:
    """agent6's state tree is single-user (transcripts, memory, run history,
    secrets), so it is created 0700 and other local users cannot traverse in --
    the files inside then need no per-file mode. Only what agent6 creates: a
    pre-existing ancestor keeps its own mode."""
    outer = tmp_path / "pre"
    outer.mkdir(mode=0o755)
    leaf = outer / "state" / "agent6" / "repo-x" / "sessions" / "runs" / "s1"
    paths.mkdir_for_real_user(leaf)
    for d in (outer / "state" / "agent6", outer / "state" / "agent6" / "repo-x", leaf):
        assert (d.stat().st_mode & 0o777) == 0o700, d
    assert (outer.stat().st_mode & 0o777) == 0o755  # pre-existing, untouched


def test_repo_id_distinguishes_paths(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert paths.repo_id(a) != paths.repo_id(b)


def test_repo_id_separates_paths_that_flatten_alike(tmp_path: Path) -> None:
    """`/a/b/c` and `/a/b-c` both flatten to `a-b-c`. Sharing one state dir
    between two real workspaces is worse than an unreadable name, so the hash
    has to separate them."""
    nested = tmp_path / "b" / "c"
    nested.mkdir(parents=True)
    dashed = tmp_path / "b-c"
    dashed.mkdir()
    assert paths.repo_id(nested) != paths.repo_id(dashed)


@pytest.mark.parametrize(
    "segment",
    [
        "segment",  # ASCII, 1 byte per char
        "日本語のディレクトリ名",  # 3 bytes per char
        "🚀🚀🚀🚀🚀",  # 4 bytes per char
        "ünïcödé-àccénts",  # 2 bytes per char
    ],
)
def test_repo_id_stays_a_usable_directory_name(tmp_path: Path, segment: str) -> None:
    """The filesystem limit is 255 BYTES per component. Capping CHARACTERS gave
    a 271-byte name for a CJK path, and every state-dir command died with an
    unhandled ENAMETOOLONG."""
    # Rooted at `/`: under `tmp_path` the ASCII prefix would take the whole
    # head cut, leaving only the tail cut inside a multi-byte segment.
    deep = Path("/", *[f"{segment}{i}" for i in range(30)])
    rid = paths.repo_id(deep)
    assert len(rid.encode()) < 255
    # Every kept character comes from the path, a separator or the hex tail:
    # a character split at the byte cut would decode to something else.
    assert set(rid) <= set(str(deep.resolve()) + "-0123456789abcdef")
    (tmp_path / rid).mkdir()  # the real filesystem accepts it


def test_state_base_uses_xdg_when_not_sudo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert paths.state_base() == tmp_path / "xdg" / "agent6"


def test_data_dir_follows_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "d"))
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    assert paths.data_dir() == tmp_path / "d" / "agent6"


def test_data_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    home = paths.effective_user().home
    assert paths.data_dir() == home / ".local" / "share" / "agent6"


def test_effective_user_resolves_sudo(monkeypatch: pytest.MonkeyPatch) -> None:
    real_uid = os.getuid()
    real_gid = os.getgid()
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", str(real_uid))
    monkeypatch.setenv("SUDO_GID", str(real_gid))
    monkeypatch.setenv("SUDO_USER", "alice")
    user = paths.effective_user()
    assert user.via_sudo is True
    assert user.uid == real_uid
    assert user.gid == real_gid
    assert user.name == "alice"


def test_effective_user_non_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.delenv("SUDO_UID", raising=False)
    user = paths.effective_user()
    assert user.via_sudo is False


def test_root_optin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT6_ALLOW_ROOT", raising=False)
    assert paths.root_optin_enabled(False) is False
    assert paths.root_optin_enabled(True) is True
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "1")
    assert paths.root_optin_enabled(False) is True
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "0")
    assert paths.root_optin_enabled(False) is False
    # "1" alone opts in, as the sandbox-disable variable takes it: any other
    # truthy-looking value ("yes", "x") opted in before.
    monkeypatch.setenv("AGENT6_ALLOW_ROOT", "yes")
    assert paths.root_optin_enabled(False) is False


def test_mkdir_for_real_user_hands_back_created_ancestors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under sudo, every directory the call CREATES is handed back to the real
    operator: chowning only the deepest one left a root-owned state/config
    BASE that no later non-root process could create a sibling in. Directories
    that already existed are never touched."""
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", "1234")
    monkeypatch.setenv("SUDO_GID", "1234")
    chowned: list[Path] = []

    def _record(*a: object) -> None:
        chowned.append(Path(str(a[0])))

    def _record_at(target: object, _uid: int, _gid: int, **kw: object) -> None:
        chowned.append(Path(f"/proc/self/fd/{kw['dir_fd']}").readlink() / str(target))

    monkeypatch.setattr(os, "lchown", _record)
    monkeypatch.setattr(os, "chown", _record_at)
    base = tmp_path / "existing"
    base.mkdir()
    target = base / "agent6" / "repo-abc"
    paths.mkdir_for_real_user(target)
    assert target.is_dir()
    assert base / "agent6" in chowned  # the created ancestor is handed back
    assert target in chowned
    assert base not in chowned  # pre-existing dirs are never rechowned
    # Nothing missing: the handover still covers the path itself (the
    # behavior of the per-site mkdir+chown pairs this primitive replaces).
    chowned.clear()
    paths.mkdir_for_real_user(target)
    assert chowned == [target]


def test_chown_to_real_user_is_noop_when_not_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    f = tmp_path / "x"
    f.write_text("hi", encoding="utf-8")
    # Must not raise and must not attempt a chown.
    called: list[object] = []

    def _fake_lchown(*a: object) -> None:
        called.append(a)

    monkeypatch.setattr(os, "lchown", _fake_lchown)
    paths.chown_to_real_user(f)
    assert called == []


def test_a_chown_never_resolves_a_symlink_swapped_in_mid_walk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under sudo this runs as root over trees a jailed command holds RW. The
    old walk listed paths then chowned them BY NAME, so swapping a parent
    directory for a symlink in between had root chown whatever it pointed at.
    The swap here is what a live escapee does; the assertion is on the inode
    each chown would actually land on."""
    tree = tmp_path / "state"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "file").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "file"
    secret.write_text("s", encoding="utf-8")
    secret_id = (secret.stat().st_dev, secret.stat().st_ino)

    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        paths,
        "effective_user",
        lambda: paths.RealUser(1000, 1000, "op", Path("/home/op"), True),
    )
    hit: list[tuple[int, int]] = []

    def _swap_then_record(st: os.stat_result) -> None:
        hit.append((st.st_dev, st.st_ino))
        if (tree / "sub").is_dir() and not (tree / "sub").is_symlink():
            shutil.rmtree(tree / "sub")
            (tree / "sub").symlink_to(outside, target_is_directory=True)

    def _fake_lchown(target: object, _uid: int, _gid: int) -> None:
        _swap_then_record(Path(str(target)).lstat())

    def _fake_chown(target: object, _uid: int, _gid: int, **kw: object) -> None:
        assert kw.get("follow_symlinks") is False, "a chown that follows links is the bug"
        _swap_then_record(os.stat(str(target), dir_fd=kw["dir_fd"], follow_symlinks=False))  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(os, "lchown", _fake_lchown)
    monkeypatch.setattr(os, "chown", _fake_chown)
    paths.chown_to_real_user(tree)

    assert hit, "nothing was handed over at all"
    assert secret_id not in hit, "root chowned a file outside the tree"


def test_state_is_keyed_on_the_project_not_the_directory_you_stood_in(tmp_path: Path) -> None:
    """From a subdirectory the state dir was a different, empty project: `runs`
    listed nothing, `resume` found nothing, and read_session and memory saw an
    empty history -- silently, since an empty project and a new one look the
    same."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "src" / "deep").mkdir(parents=True)
    assert paths.state_dir(repo / "src" / "deep") == paths.state_dir(repo)


def test_one_repo_is_one_project_even_when_the_repo_is_your_home(tmp_path: Path) -> None:
    """Stopping the walk at $HOME gave each subdirectory of a dotfiles repo its
    own state dir -- and its own repo.lock -- while `git -C` still resolved
    every one of them to the SAME working tree. Two runs then committed into it
    at once, which is exactly what the lock exists to prevent."""
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)
    (home / ".config" / "nvim").mkdir(parents=True)
    (home / "bin").mkdir()
    dirs = {paths.state_dir(home / ".config" / "nvim"), paths.state_dir(home / "bin")}
    assert dirs == {paths.state_dir(home)}, "one working tree must be one lock"


def test_the_filesystem_root_is_not_a_directory_named_root(tmp_path: Path) -> None:
    """`/` flattens to nothing, and the sentinel word for it was also a legal
    directory name: `/` and `/root` were one id, so a container with WORKDIR /
    shared config, runs and repo.lock with anything under /root."""
    assert paths.repo_id(Path("/")) != paths.repo_id(Path("/root"))


def test_a_worktree_is_the_project_it_is_a_worktree_of(tmp_path: Path) -> None:
    """A linked worktree's `.git` is a FILE, so an is_dir() walk would climb
    past it into whatever repo happens to be above."""
    tree = tmp_path / "wt"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n", encoding="utf-8")
    (tree / "sub").mkdir()
    assert paths.project_root(tree / "sub") == tree.resolve()


def test_a_linked_worktree_is_the_repository_it_belongs_to(tmp_path: Path) -> None:
    """`git worktree add` writes a `.git` FILE naming the repository's
    `.git/worktrees/<name>`: the worktree is that repository's project (one
    state dir, config and memory), the way a subdirectory is. A pointer at a
    directory that is gone (the pin above) keeps the worktree its own project."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(worktree), "HEAD")
    (worktree / "sub").mkdir()
    assert paths.project_root(worktree / "sub") == repo.resolve()
    assert paths.state_dir(worktree) == paths.state_dir(repo)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_outside_a_repo_the_directory_is_the_project(tmp_path: Path) -> None:
    plain = tmp_path / "notarepo"
    plain.mkdir()
    assert paths.project_root(plain) == plain.resolve()


@pytest.mark.parametrize(
    "path",
    ["/a/b/c", "/a/b-c", "/a-b/c", "/a-b-c", "/x/y", "/x-y", "/tmp", "/home/u/my-repo/sub"],
)
def test_the_id_decodes_back_to_the_path_it_names(path: str) -> None:
    """Two paths can never share a state dir, by CONSTRUCTION rather than by
    luck: the id is reversible. It used to be a flattened path plus a short
    hash, so paths that flatten alike were separated by 24 bits -- brute-forced
    in 11 seconds, after which one project read another's config, runs and
    transcripts."""
    rid = paths.repo_id(Path(path))
    flat, tag = rid.rsplit("-", 1)
    # The name fixes the bit LENGTH, which is what makes leading zeros safe.
    marks = bin(int(tag, 16))[2:].zfill(flat.count("-"))
    out, seen = [], 0
    for ch in flat:
        if ch == "-":
            out.append("/" if marks[seen] == "1" else "-")
            seen += 1
        else:
            out.append(ch)
    assert "/" + "".join(out) == path
    assert seen == flat.count("-"), "the tag describes exactly the dashes in the name"


def test_the_common_case_carries_no_hash_at_all(tmp_path: Path) -> None:
    """A hash is unreadable and, here, unnecessary: the tag is 1-4 characters
    and means something."""
    assert paths.repo_id(Path("/home/u/agent6")) == "home-u-agent6-3"
    assert paths.repo_id(Path("/tmp")) == "tmp-0"


def test_repo_root_of_id_inverts_repo_id() -> None:
    """`agent6 ps` decodes state-dir names back to directories: the inverse
    must round-trip every dash/slash mix, reject junk names, and reject a
    candidate that does not re-encode identically (the elided-hash form)."""
    from agent6.paths import repo_id, repo_root_of_id

    for path in ("/a/b/c", "/a/b-c", "/a-b-c", "/x---y/z-", "/tmp/a--b", "/"):
        rid = repo_id(Path(path))
        assert repo_root_of_id(rid) == Path(path), path
    assert repo_root_of_id("not-a-tag-zz") is None
    assert repo_root_of_id("plain-file") is None
    long = "/" + "/".join(["seg"] * 80)
    assert repo_root_of_id(repo_id(Path(long))) is None  # elided form: not reversible


def test_cmd_ps_lists_live_sessions_with_decoded_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One row per LIVE session across every repo state dir: the decoded
    directory (so the operator can cd there and attach), id, mode, status,
    pid; a dead session never lists."""
    import json
    import os

    from agent6.paths import repo_id
    from agent6.ui.cli import ps_cmd

    base = tmp_path / "state"
    repo = tmp_path / "proj"
    repo.mkdir()
    live = base / repo_id(repo) / "sessions" / "runs" / "brave-fox-AAAAAA"
    live.mkdir(parents=True)
    (live / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (live / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    dead = base / repo_id(repo) / "sessions" / "runs" / "dead-oak-BBBBBB"
    dead.mkdir(parents=True)
    (dead / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ps_cmd, "state_base", lambda: base)
    assert ps_cmd.cmd_ps() == 0
    out = capsys.readouterr().out
    assert "brave-fox-AAAAAA" in out and str(repo) in out.replace("~", str(Path.home()))
    assert "dead-oak-BBBBBB" not in out
    assert "agent6 attach" in out

    # An elided-hash id (a path past the byte budget) is not reversible: the
    # directory cell says so instead of offering a state-dir name to cd into.
    long_repo = tmp_path / ("q" * 200)
    long_repo.mkdir()
    elided = base / repo_id(long_repo) / "sessions" / "runs" / "long-elm-EEEEEE"
    elided.mkdir(parents=True)
    (elided / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (elided / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    assert ps_cmd.cmd_ps() == 0
    out = capsys.readouterr().out
    assert "? (" in out and "directory not recoverable" in out

    monkeypatch.setattr(ps_cmd, "state_base", lambda: tmp_path / "empty")
    assert ps_cmd.cmd_ps() == 0
    assert "no live agent6 sessions." in capsys.readouterr().out
