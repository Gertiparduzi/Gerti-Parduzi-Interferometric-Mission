"""Safely fast-forward the existing local feature branch from a delivery bundle.

Standard library only. No push, checkout, stash or environment changes. A local
empty random_search.py placeholder is backed up before clearing its conflict.
Other work is preserved; incompatible states stop with an explicit error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4


BASE_COMMIT = "6c5b3e76a6f55b28429026b76f108f788c5e9d9e"
SOURCE_BRANCH = "work/phases-4-6"
PREPARED_REF = "refs/remotes/phase46-delivery/prepared"
PLACEHOLDER = "src/random_search.py"


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", "-C", str(repo), *arguments], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result


def integrate(
    repository: Path,
    bundle: Path,
    *,
    expected_branch: str = "feat/random-search",
    base_commit: str = BASE_COMMIT,
) -> dict:
    repository, bundle = Path(repository).resolve(), Path(bundle).resolve()
    if not bundle.is_file():
        raise ValueError(f"Bundle not found: {bundle}")
    branch = git(repository, "branch", "--show-current").stdout.strip()
    if branch != expected_branch:
        raise ValueError(f"Expected branch {expected_branch!r}; active branch is {branch!r}.")
    git(repository, "cat-file", "-e", base_commit + "^{commit}")
    git(repository, "bundle", "verify", str(bundle))
    git(repository, "fetch", str(bundle), f"refs/heads/{SOURCE_BRANCH}:{PREPARED_REF}")
    target = git(repository, "rev-parse", PREPARED_REF).stdout.strip()
    head = git(repository, "rev-parse", "HEAD").stdout.strip()
    if git(repository, "merge-base", "--is-ancestor", target, head, check=False).returncode == 0:
        return {"branch": branch, "head": head, "already_integrated": True, "backup": None}
    if git(repository, "merge-base", "--is-ancestor", base_commit, target, check=False).returncode:
        raise ValueError("The delivery does not descend from the expected foundation.")
    if git(repository, "merge-base", "--is-ancestor", head, target, check=False).returncode:
        raise ValueError("Local commits diverge from the delivery; no automatic merge was attempted.")

    unstaged = set(filter(None, git(repository, "diff", "--name-only", "-z").stdout.split("\0")))
    staged = set(filter(None, git(repository, "diff", "--cached", "--name-only", "-z").stdout.split("\0")))
    if (unstaged | staged) - {PLACEHOLDER}:
        raise ValueError("Other tracked changes are present. Preserve/review them before integration.")
    placeholder = repository / PLACEHOLDER
    if placeholder.is_symlink():
        raise ValueError("The placeholder is a symlink; inspect it manually.")
    if placeholder.exists() and (not placeholder.is_file() or placeholder.stat().st_size != 0):
        raise ValueError("random_search.py is not empty; it was not overwritten.")
    if (PLACEHOLDER in staged or PLACEHOLDER in unstaged) and git(
        repository, "cat-file", "-e", f"HEAD:{PLACEHOLDER}", check=False
    ).returncode == 0:
        raise ValueError("random_search.py already exists in HEAD; inspect the local changes manually.")

    added = git(repository, "diff", "--name-only", "--diff-filter=A", "-z", "HEAD", target).stdout
    for path in filter(None, added.split("\0")):
        if path != PLACEHOLDER and (repository / path).exists():
            raise ValueError(f"A local file conflicts with the delivery: {path}. It was not overwritten.")

    backup = None
    if placeholder.exists():
        backup = bundle.parent / "local-backups" / f"random_search-{uuid4().hex}.py"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(placeholder, backup)
        if PLACEHOLDER in staged:
            git(repository, "restore", "--staged", "--", PLACEHOLDER)
        if placeholder.stat().st_size != 0:
            raise ValueError("The placeholder changed during integration; it was not removed.")
        placeholder.unlink()
    try:
        git(repository, "merge", "--ff-only", PREPARED_REF)
    except RuntimeError:
        if backup is not None and not placeholder.exists():
            shutil.copy2(backup, placeholder)
            if PLACEHOLDER in staged:
                git(repository, "add", "--", PLACEHOLDER)
        raise
    return {"branch": branch, "head": target, "already_integrated": False,
            "backup": str(backup) if backup else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-branch", default="feat/random-search")
    parser.add_argument("--base-commit", default=BASE_COMMIT)
    args = parser.parse_args()
    try:
        result = integrate(args.repo, args.bundle, expected_branch=args.expected_branch,
                           base_commit=args.base_commit)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"Integration stopped: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
