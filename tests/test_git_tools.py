"""Tests for the Git inspection tool (``lightspeed_cli.tools.git_tools``).

Learning notes
--------------
Three kinds of tests, each catching different mistakes:

1. **Parser tests** feed samples of real git output (copied from git 2.43) to
   the pure parsing functions. Fast, and no git needed.
2. **Mocked tests** replace ``subprocess.run`` with a fake, to simulate
   situations that are hard to create for real: git not installed, a command
   hanging, strange errors. They also check the *security rules* (list
   arguments, no shell, a timeout) by inspecting how the fake was called.
3. **Real-repository tests** run actual git in a temporary folder. Mocks only
   prove our code handles what we *think* git prints; these prove it handles
   what git *really* prints. They are skipped if git is not installed.
"""

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from lightspeed_cli.tools import registry
from lightspeed_cli.tools.git_tools import (
    MAX_FILES_PER_CATEGORY,
    GitError,
    _parse_log,
    _parse_shortstat,
    _parse_status_v2,
    inspect_git_status,
)

RUN = "lightspeed_cli.tools.git_tools.subprocess.run"

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git missing")


# --------------------------------------------------------------------------- #
# 1. Parser tests
# --------------------------------------------------------------------------- #
OID = "af229668279bbe67b382214914a7d24e3f2b6b47"
MODE = "100644 100644 100644"
HASHES = f"{OID} {OID}"


def nul_join(*entries: str) -> str:
    """Build -z output: every entry is followed by a NUL byte."""
    return "".join(f"{entry}\0" for entry in entries)


def test_parse_status_of_new_repository() -> None:
    status = _parse_status_v2(nul_join("# branch.oid (initial)", "# branch.head main"))

    assert status == {
        "branch": "main",
        "head": None,
        "upstream": None,
        "ahead": None,
        "behind": None,
        "staged": [],
        "unstaged": [],
        "untracked": [],
        "conflicted": [],
        "truncated": False,
    }


def test_parse_status_with_every_kind_of_change() -> None:
    output = nul_join(
        f"# branch.oid {OID}",
        "# branch.head feature/login",
        "# branch.upstream origin/feature/login",
        "# branch.ab +2 -1",
        "# stash 3",  # headers we do not use (or future ones) are ignored
        f"1 .M N... {MODE} {HASHES} src/app.py",
        f"1 A. N... {MODE} {HASHES} docs/new file.md",  # space in the name
        f"1 MD N... {MODE} {HASHES} both.txt",  # staged AND unstaged
        f"2 R. N... {MODE} {HASHES} R100 renamed.py",
        "original.py",  # a rename's old path is the *next* entry
        f"u UU N... 100644 100644 100644 100644 {OID} {OID} {OID} conflict.txt",
        "? notes.txt",
        "! ignored.log",  # ignored files are not reported
    )

    status = _parse_status_v2(output)

    assert status["branch"] == "feature/login"
    assert status["head"] == "af22966"
    assert status["upstream"] == "origin/feature/login"
    assert (status["ahead"], status["behind"]) == (2, 1)
    assert status["staged"] == [
        {"path": "docs/new file.md", "change": "added"},
        {"path": "both.txt", "change": "modified"},
        {"path": "renamed.py", "change": "renamed", "previous_path": "original.py"},
    ]
    assert status["unstaged"] == [
        {"path": "src/app.py", "change": "modified"},
        {"path": "both.txt", "change": "deleted"},
    ]
    assert status["untracked"] == ["notes.txt"]
    assert status["conflicted"] == ["conflict.txt"]
    assert status["truncated"] is False


def test_parse_status_detached_head() -> None:
    status = _parse_status_v2(
        nul_join(f"# branch.oid {OID}", "# branch.head (detached)")
    )

    assert status["branch"] is None
    assert status["head"] == "af22966"


def test_parse_status_truncates_long_lists() -> None:
    untracked = [f"? file{i}.txt" for i in range(MAX_FILES_PER_CATEGORY + 10)]

    status = _parse_status_v2(nul_join(*untracked))

    assert len(status["untracked"]) == MAX_FILES_PER_CATEGORY
    assert status["truncated"] is True


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("", {"files": 0, "insertions": 0, "deletions": 0}),
        (
            " 1 file changed, 1 insertion(+)\n",
            {"files": 1, "insertions": 1, "deletions": 0},
        ),
        (
            " 3 files changed, 10 insertions(+), 2 deletions(-)\n",
            {"files": 3, "insertions": 10, "deletions": 2},
        ),
        (
            " 2 files changed, 5 deletions(-)\n",
            {"files": 2, "insertions": 0, "deletions": 5},
        ),
    ],
    ids=["no-changes", "insertions-only", "everything", "deletions-only"],
)
def test_parse_shortstat(output: str, expected: dict[str, int]) -> None:
    assert _parse_shortstat(output) == expected


def test_parse_log() -> None:
    output = nul_join(
        "af22966\x1fAda Lovelace\x1f2026-10-21T22:58:36+02:00\x1ffirst | subject, x",
        "\n19ab95b\x1fBob\x1f2026-10-20T00:48:46+02:00\x1fInitial commit",
    )

    assert _parse_log(output) == [
        {
            "hash": "af22966",
            "author": "Ada Lovelace",
            "date": "2026-10-21T22:58:36+02:00",
            "subject": "first | subject, x",
        },
        {
            "hash": "19ab95b",
            "author": "Bob",
            "date": "2026-10-20T00:48:46+02:00",
            "subject": "Initial commit",
        },
    ]


# --------------------------------------------------------------------------- #
# 2. Mocked subprocess tests
# --------------------------------------------------------------------------- #
def completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    """A fake result, shaped exactly like what subprocess.run returns."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_git_not_installed(mocker: MockerFixture) -> None:
    mocker.patch(RUN, side_effect=FileNotFoundError("git"))

    result = registry.execute_sync("inspect_git_status")

    assert result.error == "GitError: git is not installed or not on PATH"


def test_git_timeout(mocker: MockerFixture) -> None:
    mocker.patch(RUN, side_effect=subprocess.TimeoutExpired(cmd=["git"], timeout=5))

    with pytest.raises(GitError, match="'git rev-parse' timed out after 5 seconds"):
        inspect_git_status()


def test_not_a_repository(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch(
        RUN,
        return_value=completed(
            stderr="fatal: not a git repository (or any of the parent directories)",
            returncode=128,
        ),
    )

    assert inspect_git_status(str(tmp_path)) == {
        "is_git_repo": False,
        "path": str(tmp_path),
    }


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (
            "fatal: detected dubious ownership in repository at '/srv/x'\nmore",
            "'git rev-parse' failed: fatal: detected dubious ownership in "
            "repository at '/srv/x'",
        ),
        ("", "'git rev-parse' failed: exit code 1"),
    ],
    ids=["with-message", "without-message"],
)
def test_other_git_failures_are_errors(
    mocker: MockerFixture, stderr: str, expected: str
) -> None:
    mocker.patch(RUN, return_value=completed(stderr=stderr, returncode=1))

    with pytest.raises(GitError) as exc_info:
        inspect_git_status()

    assert str(exc_info.value) == expected


def test_git_is_called_safely(mocker: MockerFixture, tmp_path: Path) -> None:
    run = mocker.patch(
        RUN,
        return_value=completed(stderr="fatal: not a git repository", returncode=128),
    )

    inspect_git_status(str(tmp_path))

    # call_args holds the positional and keyword arguments of the last call.
    args, kwargs = run.call_args
    command = args[0]
    assert isinstance(command, list)  # a list, never a shell string
    assert command[:3] == ["git", "-c", "core.fsmonitor=false"]
    assert "shell" not in kwargs  # shell defaults to False
    assert kwargs["timeout"] == 5.0
    assert kwargs["cwd"] == tmp_path  # the user's path is only ever the cwd
    assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert kwargs["env"]["LC_ALL"] == "C"


@pytest.mark.parametrize(
    ("make_path", "error"),
    [
        (lambda tmp: tmp / "missing", FileNotFoundError),
        (lambda tmp: tmp / "file.txt", NotADirectoryError),
    ],
    ids=["missing", "file"],
)
def test_invalid_path(
    tmp_path: Path,
    make_path: Callable[[Path], Path],
    error: type[Exception],
) -> None:
    (tmp_path / "file.txt").write_text("x", encoding="utf-8")

    with pytest.raises(error):
        inspect_git_status(str(make_path(tmp_path)))


# --------------------------------------------------------------------------- #
# 3. Real-repository tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make git behave the same on every machine.

    Your personal ~/.gitconfig (aliases, default branch, hidden untracked
    files, ...) must not change test results, and commits need an author.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test User")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "test@example.com")


@pytest.fixture
def repo(tmp_path: Path, git_env: None) -> Path:
    """A fresh, empty git repository on branch "main"."""
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "--quiet", "--initial-branch=main")
    return path


def git(repo: Path, *args: str) -> None:
    """Run a git command in the test repository (check=True: fail loudly)."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def commit(repo: Path, filename: str, content: str, message: str) -> None:
    (repo / filename).write_text(content, encoding="utf-8")
    git(repo, "add", filename)
    git(repo, "commit", "--quiet", "-m", message)


@requires_git
def test_real_non_repository(tmp_path: Path, git_env: None) -> None:
    assert inspect_git_status(str(tmp_path)) == {
        "is_git_repo": False,
        "path": str(tmp_path),
    }


@requires_git
def test_real_empty_repository(repo: Path) -> None:
    report = inspect_git_status(str(repo))

    assert report["is_git_repo"] is True
    assert report["root"] == str(repo)
    assert report["branch"] == "main"
    assert report["head"] is None
    assert report["recent_commits"] == []


@requires_git
def test_real_clean_repository(repo: Path) -> None:
    commit(repo, "a.txt", "one\n", "First commit")

    report = inspect_git_status(str(repo))

    assert report["is_git_repo"] is True
    assert report["head"] is not None
    assert report["staged"] == report["unstaged"] == []
    assert report["untracked"] == []
    assert report["diff"]["staged"]["files"] == 0
    assert [c["subject"] for c in report["recent_commits"]] == ["First commit"]
    assert report["recent_commits"][0]["author"] == "Test User"


@requires_git
def test_real_dirty_repository(repo: Path) -> None:
    commit(repo, "a.txt", "one\n", "First commit")
    (repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")  # unstaged edit
    (repo / "b.txt").write_text("new\n", encoding="utf-8")
    git(repo, "add", "b.txt")  # staged new file
    (repo / "notes with spaces.txt").write_text("x", encoding="utf-8")  # untracked

    # A subdirectory works too: git searches upwards for the repository.
    (repo / "sub").mkdir()
    report = inspect_git_status(str(repo / "sub"))

    assert report["is_git_repo"] is True
    assert report["root"] == str(repo)
    assert report["staged"] == [{"path": "b.txt", "change": "added"}]
    assert report["unstaged"] == [{"path": "a.txt", "change": "modified"}]
    assert report["untracked"] == ["notes with spaces.txt"]
    assert report["diff"] == {
        "staged": {"files": 1, "insertions": 1, "deletions": 0},
        "unstaged": {"files": 1, "insertions": 1, "deletions": 0},
    }


@requires_git
def test_real_max_commits(repo: Path) -> None:
    for number in range(5):
        commit(repo, "a.txt", f"{number}\n", f"Commit {number}")

    newest_two = inspect_git_status(str(repo), max_commits=2)
    none = inspect_git_status(str(repo), max_commits=0)

    assert newest_two["is_git_repo"] is True
    assert [c["subject"] for c in newest_two["recent_commits"]] == [
        "Commit 4",
        "Commit 3",
    ]
    assert none["is_git_repo"] is True
    assert none["recent_commits"] == []


@requires_git
def test_real_repository_through_the_registry(repo: Path) -> None:
    commit(repo, "a.txt", "one\n", "First commit")

    result = registry.execute_sync("inspect_git_status", {"path": str(repo)})

    assert result.status == "success"
    assert isinstance(result.output, dict)
    assert result.output["branch"] == "main"
