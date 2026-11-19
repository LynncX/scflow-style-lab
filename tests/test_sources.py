"""Exercise checkout/patch safety entirely with a tiny local Git repository."""
import difflib
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from sources import validate_source

SPEC = importlib.util.spec_from_file_location("bootstrap", Path(__file__).resolve().parents[1] / "scripts/bootstrap.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class SourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="source safety ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.local = self.root / "local"
        source = self.local / "upstream"
        source.mkdir(parents=True)
        def git(*args):
            return subprocess.check_output(["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL).strip()
        git("init")
        old, new = "first\nold\nlast\n", "first\nnew\nlast\n"
        (source / "file.txt").write_text(old)
        git("add", "file.txt")
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")
        self.spec = {"directory": "upstream", "commit": git("rev-parse", "HEAD"), "url": "unused",
                     "patch": "patches/change.patch", "patched_files_sha256": {"file.txt": hashlib.sha256(new.encode()).hexdigest()}}
        (self.root / "manifests").mkdir()
        (self.root / "manifests/upstreams.json").write_text(json.dumps({"adain": self.spec}))
        (self.root / "patches").mkdir()
        (self.root / self.spec["patch"]).write_text("".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                                                        fromfile="a/file.txt", tofile="b/file.txt")))
        self.target = self.root / "third_party/upstream"

    def test_prepare_idempotent_and_shared_source_untouched(self):
        first = BOOTSTRAP.prepare(self.root, "adain", self.spec, self.local)
        self.assertEqual(first, BOOTSTRAP.prepare(self.root, "adain", self.spec, self.local))
        self.assertIn("old", (self.local / "upstream/file.txt").read_text())
        self.assertIn("new", (self.target / "file.txt").read_text())

    def test_modified_checkout_not_overwritten(self):
        BOOTSTRAP.prepare(self.root, "adain", self.spec, self.local)
        (self.target / "file.txt").write_text("user changes")
        with self.assertRaises(ValueError):
            BOOTSTRAP.prepare(self.root, "adain", self.spec, self.local)
        self.assertEqual((self.target / "file.txt").read_text(), "user changes")

    def test_untracked_changes_rejected(self):
        BOOTSTRAP.prepare(self.root, "adain", self.spec, self.local)
        (self.target / "extra.txt").write_text("keep")
        with self.assertRaises(ValueError):
            validate_source(self.root, "adain", self.target)
