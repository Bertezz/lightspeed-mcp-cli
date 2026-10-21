"""Built-in tool that inspects a local Git repository (read-only).

Learning notes
--------------
**Running other programs safely.** Python starts other programs with
``subprocess.run``. The rules followed here:

* **Never ``shell=True``.** With a shell, the command is one string that the
  shell interprets, so text like ``; rm -rf ~`` inside a value becomes a second
  command (*shell injection*). Passing a ``list[str]`` runs the program
  directly; every list item is one argument and is never interpreted.
* **No user input in the argument list.** The only user-controlled value,
  ``path``, is passed as ``cwd=`` (the working directory). So even a "path"
  such as ``--upload-pack=evil`` can never be read by git as an option.
* **Always a timeout.** ``subprocess.run(timeout=...)`` *kills* the program if
  it takes too long. That is stronger than the registry's timeout, which can
  stop waiting for a thread but cannot stop the thread itself.
* **``check=False``.** We inspect the exit code ourselves (0 = success) to
  produce clean error messages, instead of letting a CalledProcessError escape.

**Parsing program output safely.** We ask git for *machine-readable* output:

* ``--porcelain=v2``: a stable format meant for scripts, guaranteed not to
  change between git versions (the normal output is for humans and may).
* ``-z``: entries are separated by NUL bytes (``\\0``). Filenames can contain
  spaces and even newlines, but never NUL, so splitting on NUL is always safe.
* ``%x1f`` in the ``git log`` format is the ASCII "unit separator" character,
  which never appears in normal text, so we can split fields on it safely.

**Untrusted repositories.** A repository's own ``.git/config`` can make git run
programs automatically (``core.fsmonitor``, external diff drivers). Since the AI
may point this tool at any folder, those features are switched off.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict, TypeVar

from pydantic import Field

from lightspeed_cli.tools.base import registry

# A type variable: "some type, the same everywhere it appears in a signature".
# (Python 3.12+ has a shorter syntax, def _cap[T](...), but we support 3.11.)
T = TypeVar("T")

GIT_COMMAND_TIMEOUT = 5.0
"""Seconds each git command may run before it is killed."""

MAX_FILES_PER_CATEGORY = 50
"""Longer file lists are cut off, so huge changes do not flood the AI."""

# Every git command starts with these. ``-c key=value`` overrides a config
# setting for this one command: here, never run a file-system-monitor program
# configured by the repository.
_GIT_BASE_COMMAND = ["git", "-c", "core.fsmonitor=false"]

# Never run external diff programs or text converters the repository configures.
_SAFE_DIFF_FLAGS = ["--no-ext-diff", "--no-textconv"]

# %h short hash, %an author name, %aI author date (ISO 8601), %s subject line.
# The author *email* (%ae) is left out on purpose: tool output is sent to the AI
# provider, so we only send what is needed.
_LOG_FORMAT = "%h%x1f%an%x1f%aI%x1f%s"

# XY status letters from `git status --porcelain=v2`, as readable words.
_CHANGE_NAMES = {
    "M": "modified",
    "T": "type changed",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
}

_SHORTSTAT_FILES = re.compile(r"(\d+) files? changed")
_SHORTSTAT_INSERTIONS = re.compile(r"(\d+) insertions?\(\+\)")
_SHORTSTAT_DELETIONS = re.compile(r"(\d+) deletions?\(-\)")


class GitError(RuntimeError):
    """A git command could not be run or failed.

    The registry turns it into an error result like "GitError: git is not
    installed or not on PATH", so the message is written for the AI to read.
    """


# --------------------------------------------------------------------------- #
# Output shapes
# --------------------------------------------------------------------------- #
class FileChange(TypedDict):
    path: str
    change: str
    # NotRequired: the key is only present for renamed/copied files.
    previous_path: NotRequired[str]


class DiffStat(TypedDict):
    files: int
    insertions: int
    deletions: int


class DiffSummary(TypedDict):
    staged: DiffStat
    unstaged: DiffStat


class CommitInfo(TypedDict):
    hash: str
    author: str
    date: str
    subject: str


class StatusInfo(TypedDict):
    branch: str | None  # None when HEAD is detached
    head: str | None  # None in a new repository without commits
    upstream: str | None  # e.g. "origin/main"; None if not tracking anything
    ahead: int | None  # commits we have that upstream does not
    behind: int | None  # commits upstream has that we do not
    staged: list[FileChange]  # changes added with `git add`
    unstaged: list[FileChange]  # changes not yet added
    untracked: list[str]  # files git does not know about yet
    conflicted: list[str]  # files with merge conflicts
    truncated: bool  # True if any list was cut at MAX_FILES_PER_CATEGORY


class NotARepository(TypedDict):
    is_git_repo: Literal[False]
    path: str


class RepositoryStatus(StatusInfo):
    # Inheriting from a TypedDict adds these keys to all of StatusInfo's.
    is_git_repo: Literal[True]
    root: str
    diff: DiffSummary
    recent_commits: list[CommitInfo]


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #
@registry.register(timeout=20)
def inspect_git_status(
    path: Annotated[
        str,
        Field(
            description="Directory inside the repository. Defaults to the "
            "current directory."
        ),
    ] = ".",
    max_commits: Annotated[
        int, Field(ge=0, le=20, description="How many recent commits to list.")
    ] = 3,
) -> RepositoryStatus | NotARepository:
    """Report the state of a local Git repository: current branch, upstream,
    staged/unstaged/untracked files, size of the changes and recent commits.
    """
    directory = _resolve_directory(path)

    # Step 1: are we inside a repository at all? Not being in one is a normal
    # answer, not an error, so it returns a result instead of raising.
    try:
        root = _run_git(["rev-parse", "--show-toplevel"], directory).strip()
    except GitError as exc:
        if "not a git repository" in str(exc):
            return {"is_git_repo": False, "path": str(directory)}
        raise  # anything else (e.g. "dubious ownership") is a real problem

    # Step 2: branch, upstream and changed files, all from one command.
    status = _parse_status_v2(
        _run_git(["status", "--porcelain=v2", "--branch", "-z"], directory)
    )

    # Step 3: how big are the changes? --cached means "what is staged".
    diff: DiffSummary = {
        "staged": _parse_shortstat(
            _run_git(["diff", "--cached", "--shortstat", *_SAFE_DIFF_FLAGS], directory)
        ),
        "unstaged": _parse_shortstat(
            _run_git(["diff", "--shortstat", *_SAFE_DIFF_FLAGS], directory)
        ),
    }

    # Step 4: recent commits. A brand-new repository has none, and `git log`
    # would fail there, so skip it.
    commits: list[CommitInfo] = []
    if status["head"] is not None and max_commits > 0:
        commits = _parse_log(
            _run_git(
                ["log", f"--max-count={max_commits}", "-z", f"--format={_LOG_FORMAT}"],
                directory,
            )
        )

    return {
        "is_git_repo": True,
        "root": root,
        "branch": status["branch"],
        "head": status["head"],
        "upstream": status["upstream"],
        "ahead": status["ahead"],
        "behind": status["behind"],
        "staged": status["staged"],
        "unstaged": status["unstaged"],
        "untracked": status["untracked"],
        "conflicted": status["conflicted"],
        "truncated": status["truncated"],
        "diff": diff,
        "recent_commits": commits,
    }


# --------------------------------------------------------------------------- #
# Running git
# --------------------------------------------------------------------------- #
def _resolve_directory(path: str) -> Path:
    # expanduser() turns "~" into your home folder; resolve() makes the path
    # absolute and follows symlinks, so error messages show the real location.
    directory = Path(path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"No such directory: {path}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    return directory


def _git_environment() -> dict[str, str]:
    # Start from the current environment (so PATH, HOME, ... still work) and
    # override a few variables for predictable, side-effect-free runs.
    return {
        **os.environ,
        # Do not take the .git/index.lock that `git status` normally grabs to
        # refresh its cache; otherwise it could clash with your own git commands.
        "GIT_OPTIONAL_LOCKS": "0",
        # English messages regardless of your system language, because we
        # search error messages for "not a git repository".
        "LC_ALL": "C",
        # Never stop and wait for a password prompt.
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git(args: list[str], cwd: Path) -> str:
    """Run one git command and return its standard output.

    Raises:
        GitError: git is missing, timed out, or exited with an error.
    """
    try:
        completed = subprocess.run(
            [*_GIT_BASE_COMMAND, *args],  # a list, never a shell string
            cwd=cwd,
            capture_output=True,  # collect stdout/stderr instead of printing
            text=True,  # give us str instead of bytes
            encoding="utf-8",
            errors="replace",  # odd bytes in filenames cannot crash decoding
            timeout=GIT_COMMAND_TIMEOUT,
            check=False,  # we look at the exit code ourselves, below
            env=_git_environment(),
        )
    except FileNotFoundError as exc:
        # The operating system could not find a program called "git".
        raise GitError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        # subprocess.run has already killed git at this point.
        raise GitError(
            f"'git {args[0]}' timed out after {GIT_COMMAND_TIMEOUT:g} seconds"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = (
            stderr.splitlines()[0] if stderr else f"exit code {completed.returncode}"
        )
        raise GitError(f"'git {args[0]}' failed: {detail}")
    return completed.stdout


# --------------------------------------------------------------------------- #
# Parsing. These are "pure" functions: text in, data out, no git needed. That
# makes them easy to test with samples of real git output.
# --------------------------------------------------------------------------- #
def _parse_status_v2(output: str) -> StatusInfo:
    """Parse ``git status --porcelain=v2 --branch -z`` output.

    Entry formats (fields separated by spaces, entries by NUL)::

        # branch.oid <commit> | (initial)
        # branch.head <name> | (detached)
        # branch.upstream <upstream>
        # branch.ab +<ahead> -<behind>
        1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>              changed
        2 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <score> <path>\\0<orig>  renamed
        u <XY> <sub> <m1> <m2> <m3> <mW> <h1> <h2> <h3> <path>    conflict
        ? <path>                                                    untracked

    X is the staged state and Y the unstaged state; "." means unchanged.
    """
    branch: str | None = None
    head: str | None = None
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None
    staged: list[FileChange] = []
    unstaged: list[FileChange] = []
    untracked: list[str] = []
    conflicted: list[str] = []

    # iter() lets us pull an extra item with next(): a rename's original path
    # arrives as the *following* NUL-separated entry.
    entries = iter(output.split("\0"))
    for entry in entries:
        if not entry:
            continue
        kind = entry[0]
        if kind == "#":
            key, _, value = entry[2:].partition(" ")
            if key == "branch.oid":
                head = None if value == "(initial)" else value[:7]
            elif key == "branch.head":
                branch = None if value == "(detached)" else value
            elif key == "branch.upstream":
                upstream = value
            elif key == "branch.ab":
                ahead_text, behind_text = value.split()
                ahead = int(ahead_text.lstrip("+"))
                behind = int(behind_text.lstrip("-"))
        elif kind == "1":
            # maxsplit keeps spaces inside the path: the path is the last field.
            fields = entry.split(" ", 8)
            _add_change(fields[1], fields[8], staged, unstaged)
        elif kind == "2":
            fields = entry.split(" ", 9)
            previous = next(entries, "")
            _add_change(fields[1], fields[9], staged, unstaged, previous)
        elif kind == "u":
            conflicted.append(entry.split(" ", 10)[10])
        elif kind == "?":
            untracked.append(entry[2:])

    staged, cut_staged = _cap(staged)
    unstaged, cut_unstaged = _cap(unstaged)
    untracked, cut_untracked = _cap(untracked)
    conflicted, cut_conflicted = _cap(conflicted)
    return {
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "conflicted": conflicted,
        "truncated": cut_staged or cut_unstaged or cut_untracked or cut_conflicted,
    }


def _add_change(
    xy: str,
    path: str,
    staged: list[FileChange],
    unstaged: list[FileChange],
    previous_path: str | None = None,
) -> None:
    """Record one file in the staged and/or unstaged list.

    A file can be in both: e.g. "MM" means some edits were staged with
    ``git add`` and more edits were made afterwards.
    """
    for code, target in ((xy[0], staged), (xy[1], unstaged)):
        if code == ".":
            continue
        change: FileChange = {"path": path, "change": _CHANGE_NAMES.get(code, code)}
        if previous_path and code in ("R", "C"):
            change["previous_path"] = previous_path
        target.append(change)


def _cap(items: list[T]) -> tuple[list[T], bool]:
    """Return at most MAX_FILES_PER_CATEGORY items, and whether any were cut."""
    return items[:MAX_FILES_PER_CATEGORY], len(items) > MAX_FILES_PER_CATEGORY


def _parse_shortstat(output: str) -> DiffStat:
    """Parse ``git diff --shortstat``, e.g.
    ``" 3 files changed, 10 insertions(+), 2 deletions(-)"``.

    Parts are left out when zero ("1 file changed, 1 insertion(+)"), and the
    output is empty when there are no changes, so each number is optional.
    """

    def number(pattern: re.Pattern[str]) -> int:
        match = pattern.search(output)
        return int(match.group(1)) if match else 0

    return {
        "files": number(_SHORTSTAT_FILES),
        "insertions": number(_SHORTSTAT_INSERTIONS),
        "deletions": number(_SHORTSTAT_DELETIONS),
    }


def _parse_log(output: str) -> list[CommitInfo]:
    """Parse ``git log -z --format=%h%x1f%an%x1f%aI%x1f%s`` output."""
    commits: list[CommitInfo] = []
    for record in output.split("\0"):
        record = record.strip("\n")
        if not record:
            continue
        # maxsplit=3: a subject containing the separator would stay in one piece.
        short_hash, author, date, subject = record.split("\x1f", 3)
        commits.append(
            {"hash": short_hash, "author": author, "date": date, "subject": subject}
        )
    return commits
