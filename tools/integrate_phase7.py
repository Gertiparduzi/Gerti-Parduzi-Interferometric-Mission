"""Import phase 7 into a NEW linked worktree without touching existing work.

Standard library only; works when --repo is itself a linked worktree. No
push, reset, clean, stash, merge, or update of main/feat/random-search.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


BASE_COMMIT = '355adee126cff1ab9c4d2ece368e25042b542b7f'
SOURCE_BRANCH = 'work/advanced-optimization'


def git(repo, *args, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result


def integrate(repository, bundle, worktree, *, branch='feat/advanced-optimization',
              base_commit=BASE_COMMIT, check_only=False):
    repository, bundle, worktree = map(lambda p: Path(p).resolve(), (repository, bundle, worktree))
    if branch in ('main', 'master', 'feat/random-search'):
        raise ValueError('Choose a new phase-7 branch; protected branches are not modified.')
    git(repository, 'check-ref-format', '--branch', branch)
    if not bundle.is_file():
        raise ValueError(f'Bundle not found: {bundle}')
    if worktree == repository or repository in worktree.parents:
        raise ValueError('Choose a new sibling directory, not a directory inside the existing worktree.')
    git(repository, 'cat-file', '-e', base_commit + '^{commit}')
    git(repository, 'bundle', 'verify', str(bundle))
    heads = git(repository, 'bundle', 'list-heads', str(bundle), f'refs/heads/{SOURCE_BRANCH}').stdout.splitlines()
    if len(heads) != 1:
        raise ValueError('Bundle must expose exactly the expected phase-7 source branch.')
    target = heads[0].split()[0]
    # Fetch only objects/FETCH_HEAD, never an existing branch or a worktree file.
    git(repository, 'fetch', '--no-tags', str(bundle), f'refs/heads/{SOURCE_BRANCH}')
    if git(repository, 'merge-base', '--is-ancestor', base_commit, target, check=False).returncode:
        raise ValueError('Bundle does not descend from the authoritative phase-6 commit.')
    if worktree.exists():
        actual_root = git(worktree, 'rev-parse', '--show-toplevel', check=False)
        if actual_root.returncode == 0 and Path(actual_root.stdout.strip()).resolve() == worktree:
            same_common = Path(git(repository, 'rev-parse', '--path-format=absolute', '--git-common-dir').stdout.strip()).resolve() == Path(git(worktree, 'rev-parse', '--path-format=absolute', '--git-common-dir').stdout.strip()).resolve()
            current = git(worktree, 'branch', '--show-current').stdout.strip()
            head = git(worktree, 'rev-parse', 'HEAD').stdout.strip()
            if same_common and current == branch and head == target:
                return dict(already_integrated=True, worktree=str(worktree), branch=branch, head=target)
        raise ValueError('Destination already exists; no files were overwritten.')
    if git(repository, 'show-ref', '--verify', '--quiet', 'refs/heads/' + branch, check=False).returncode == 0:
        raise ValueError('Target branch already exists; choose a different new branch.')
    result = dict(already_integrated=False, base_commit=base_commit, head=target,
                  worktree=str(worktree), branch=branch, check_only=check_only)
    if not check_only:
        git(repository, 'worktree', 'add', '-b', branch, str(worktree), target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--branch', default='feat/advanced-optimization')
    parser.add_argument('--check-only', action='store_true',
                        help='Verify and fetch objects, but do not create a branch/worktree.')
    args = parser.parse_args()
    try:
        result = integrate(args.repo, args.bundle, args.worktree, branch=args.branch, check_only=args.check_only)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'Integration stopped: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
