"""Exercise real Git bundles and dirty linked worktrees without user data."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.integrate_phase7 import integrate


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()


class TestPhase7Integration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / 'original repo'
        self.repo.mkdir()
        git(self.repo, 'init', '-b', 'main')
        git(self.repo, 'config', 'user.name', 'Integration Fixture')
        git(self.repo, 'config', 'user.email', 'fixture@example.invalid')
        (self.repo / 'base.txt').write_text('base\n')
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-m', 'base')
        self.base = git(self.repo, 'rev-parse', 'HEAD')
        git(self.repo, 'switch', '-c', 'work/advanced-optimization')
        (self.repo / 'new.py').write_text('phase7\n')
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-m', 'phase7')
        self.head = git(self.repo, 'rev-parse', 'HEAD')
        self.bundle = self.root / 'phase7.bundle'
        git(self.repo, 'bundle', 'create', str(self.bundle), 'work/advanced-optimization', '^' + self.base)
        git(self.repo, 'switch', 'main')
        self.linked = self.root / 'existing linked worktree'
        git(self.repo, 'worktree', 'add', '-b', 'feat/random-search', str(self.linked), self.base)
        self.destination = self.root / 'new linked worktree'

    def tearDown(self):
        self.temp.cleanup()

    def run_import(self, **kwargs):
        return integrate(self.linked, self.bundle, self.destination, base_commit=self.base, **kwargs)

    def test_dirty_staged_untracked_and_ignored_files_are_preserved(self):
        (self.linked / 'base.txt').write_text('my staged edit')
        git(self.linked, 'add', 'base.txt')
        (self.linked / 'new.py').write_text('my untracked file')
        (self.linked / '.gitignore').write_text('notes.private\n')
        (self.linked / 'notes.private').write_text('my ignored notes')
        before = git(self.linked, 'status', '--porcelain')
        result = self.run_import()
        self.assertEqual(result['head'], self.head)
        self.assertEqual(git(self.linked, 'status', '--porcelain'), before)
        self.assertEqual((self.linked / 'new.py').read_text(), 'my untracked file')
        self.assertEqual((self.linked / 'base.txt').read_text(), 'my staged edit')
        self.assertEqual((self.linked / 'notes.private').read_text(), 'my ignored notes')
        self.assertEqual(git(self.repo, 'rev-parse', 'main'), self.base)
        self.assertEqual(git(self.linked, 'rev-parse', 'HEAD'), self.base)
        self.assertEqual((self.destination / 'new.py').read_text(), 'phase7\n')
        self.assertTrue((self.destination / '.git').is_file())

    def test_check_only_does_not_create_branch_or_worktree(self):
        self.assertTrue(self.run_import(check_only=True)['check_only'])
        self.assertFalse(self.destination.exists())
        self.assertEqual(git(self.repo, 'branch', '--list', 'feat/advanced-optimization'), '')

    def test_existing_destination_stops_without_overwrite(self):
        self.destination.mkdir()
        marker = self.destination / 'notes.txt'
        marker.write_text('keep me')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_import()
        self.assertEqual(marker.read_text(), 'keep me')

    def test_existing_branch_stops_without_reset(self):
        git(self.repo, 'branch', 'feat/advanced-optimization', self.base)
        with self.assertRaisesRegex(ValueError, 'branch already exists'):
            self.run_import()
        self.assertEqual(git(self.repo, 'rev-parse', 'feat/advanced-optimization'), self.base)

    def test_protected_branch_rejected(self):
        for branch in ('main', 'master', 'feat/random-search'):
            with self.assertRaisesRegex(ValueError, 'protected'):
                self.run_import(branch=branch)

    def test_idempotent_import_preserves_changes_in_new_worktree(self):
        self.run_import()
        (self.destination / 'new.py').write_text('subsequent local change')
        self.assertTrue(self.run_import()['already_integrated'])
        self.assertEqual((self.destination / 'new.py').read_text(), 'subsequent local change')

    def test_nested_destination_rejected(self):
        with self.assertRaisesRegex(ValueError, 'sibling'):
            integrate(self.linked, self.bundle, self.linked / 'nested', base_commit=self.base)


if __name__ == '__main__':
    unittest.main()
