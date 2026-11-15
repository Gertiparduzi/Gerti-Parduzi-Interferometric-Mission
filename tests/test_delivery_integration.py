"""Real temporary Git repositories model the user's staged/untracked stub."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.integrate_delivery import integrate


def git(repo, *arguments):
    result = subprocess.run(["git", "-C", str(repo), *arguments], check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()


class TestDeliveryIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source repository"
        cls.source.mkdir()
        git(cls.source, "init", "-b", "foundation")
        git(cls.source, "config", "user.name", "Test Fixture")
        git(cls.source, "config", "user.email", "fixture@example.invalid")
        (cls.source / "README.md").write_text("base\n")
        git(cls.source, "add", "README.md")
        git(cls.source, "commit", "-m", "base")
        cls.base = git(cls.source, "rev-parse", "HEAD")
        git(cls.source, "switch", "-c", "work/phases-4-6")
        (cls.source / "src").mkdir()
        (cls.source / "src/random_search.py").write_text("# completed algorithm\n")
        (cls.source / "src/genetic_algorithm.py").write_text("# another algorithm\n")
        git(cls.source, "add", "src/random_search.py", "src/genetic_algorithm.py")
        git(cls.source, "commit", "-m", "delivery")
        cls.target = git(cls.source, "rev-parse", "HEAD")
        cls.bundle = cls.root / "mission.bundle"
        git(cls.source, "bundle", "create", str(cls.bundle), "work/phases-4-6", "^" + cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=self.root)
        self.repo = Path(self.folder.name) / "user worktree"
        git(self.root, "clone", str(self.source), str(self.repo))
        git(self.repo, "switch", "-c", "feat/random-search", self.base)
        (self.repo / "src").mkdir(exist_ok=True)

    def tearDown(self):
        self.folder.cleanup()

    def deliver(self):
        return integrate(self.repo, self.bundle, base_commit=self.base)

    def test_untracked_empty_stub_and_unrelated_docs_are_preserved(self):
        (self.repo / "src/random_search.py").touch()
        (self.repo / "docs").mkdir()
        (self.repo / "docs/notes.md").write_text("my local work")
        result = self.deliver()
        self.assertEqual(git(self.repo, "branch", "--show-current"), "feat/random-search")
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.target)
        self.assertEqual(Path(result["backup"]).read_bytes(), b"")
        self.assertEqual((self.repo / "docs/notes.md").read_text(), "my local work")
        self.assertEqual((self.repo / "src/random_search.py").read_text(), "# completed algorithm\n")

    def test_staged_empty_stub_is_backed_up_and_integrated(self):
        (self.repo / "src/random_search.py").touch()
        git(self.repo, "add", "src/random_search.py")
        result = self.deliver()
        self.assertTrue(Path(result["backup"]).exists())
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")

    def test_nonempty_stub_stops_without_data_loss(self):
        path = self.repo / "src/random_search.py"
        path.write_text("my new code")
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.deliver()
        self.assertEqual(path.read_text(), "my new code")
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)

    def test_wrong_branch_stops(self):
        git(self.repo, "switch", "-c", "other-branch")
        with self.assertRaisesRegex(ValueError, "Expected branch"):
            self.deliver()
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)

    def test_other_tracked_changes_stop_and_remain_intact(self):
        (self.repo / "README.md").write_text("local change")
        (self.repo / "src/random_search.py").touch()
        with self.assertRaisesRegex(ValueError, "Other tracked changes"):
            self.deliver()
        self.assertEqual((self.repo / "README.md").read_text(), "local change")
        self.assertTrue((self.repo / "src/random_search.py").exists())

    def test_untracked_conflicting_file_is_not_overwritten(self):
        (self.repo / "src/genetic_algorithm.py").write_text("my GA")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            self.deliver()
        self.assertEqual((self.repo / "src/genetic_algorithm.py").read_text(), "my GA")

    def test_divergent_local_commit_stops_without_reset(self):
        git(self.repo, "config", "user.name", "Local Fixture")
        git(self.repo, "config", "user.email", "local@example.invalid")
        (self.repo / "README.md").write_text("own commit")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-m", "own work")
        before = git(self.repo, "rev-parse", "HEAD")
        with self.assertRaisesRegex(ValueError, "diverge"):
            self.deliver()
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), before)

    def test_repeat_integration_is_idempotent(self):
        self.deliver()
        self.assertTrue(self.deliver()["already_integrated"])
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.target)


if __name__ == "__main__":
    unittest.main()
