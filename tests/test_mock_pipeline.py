"""Offline protocol tests: no model dependencies or installation."""
import json
import struct
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from runner import run, apply_overrides
from backends.mock import MockBackend, mock_png


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mock pipeline ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "content image.png").write_bytes(mock_png(2))
        (self.root / "style.png").write_bytes(mock_png(3))
        self.samples = [{"sample_id": sid, "content_path": "content image.png", "style_path": "style.png"}
                        for sid in ("second", "first")]
        self.manifest = self.root / "pairs.jsonl"
        self.cfg = dict(run_name="test", backend="mock", method="adain", image_size=512,
                        manifest=str(self.manifest), data_root=str(self.root))
        self.config = self.root / "config.json"
        self.out = self.root / "output"
        self.save()

    def save(self):
        self.config.write_text(json.dumps(self.cfg))
        self.manifest.write_text("".join(json.dumps(s) + "\n" for s in self.samples))

    def run_pipeline(self, resume=False):
        return run(self.config, out_dir_override=str(self.out), resume=resume)

    def records(self):
        return [json.loads(s) for s in (self.out / "records.jsonl").read_text().splitlines()]

    def test_four_methods_exact_pairs_and_resume(self):
        for method in ("scflow", "adain", "stytr2", "styleid"):
            with self.subTest(method=method):
                self.cfg["method"] = method
                self.out = self.root / method
                self.save()
                self.run_pipeline()
                records = self.records()
                self.assertEqual([r["sample_id"] for r in records], ["second", "first"])
                self.assertTrue(all(r["simulated"] and r["method"] == method for r in records))
                image = self.out / "predictions/second/second.png"
                self.assertEqual(struct.unpack(">II", image.read_bytes()[16:24]), (512, 512))
                self.assertEqual(bool(list(image.parent.glob("clip*.json"))), method == "scflow")
                with patch.object(MockBackend, "infer", side_effect=AssertionError("must skip")):
                    self.run_pipeline(resume=True)
                self.assertEqual(len(self.records()), 2)

    def test_failed_sample_retried(self):
        original = MockBackend.infer
        def fail(backend, sample, cfg, out):
            if sample["sample_id"] == "second":
                raise RuntimeError("temporary failure")
            return original(backend, sample, cfg, out)
        with patch.object(MockBackend, "infer", fail):
            self.run_pipeline()
        self.run_pipeline(resume=True)
        self.assertEqual([r["status"] for r in self.records()], ["error", "ok", "ok"])

    def test_interrupted_run_persists_first_sample(self):
        original = MockBackend.infer
        def interrupt(backend, sample, cfg, out):
            if sample["sample_id"] == "first":
                raise KeyboardInterrupt()
            return original(backend, sample, cfg, out)
        with patch.object(MockBackend, "infer", interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_pipeline()
        self.assertEqual(len(self.records()), 1)
        with (self.out / "records.jsonl").open("a") as f:
            f.write('{"sample_id":')
        self.run_pipeline(resume=True)
        self.assertEqual(len(self.records()), 2)

    def test_missing_image_retried(self):
        self.run_pipeline()
        (self.out / "predictions/second/second.png").unlink()
        self.run_pipeline(resume=True)
        self.assertEqual(len(self.records()), 3)

    def test_changed_config_rejected(self):
        self.run_pipeline()
        self.cfg["seed"] = 99
        self.save()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_pipeline(resume=True)

    def test_changed_input_rejected(self):
        self.run_pipeline()
        (self.root / "style.png").write_bytes(mock_png(4))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_pipeline(resume=True)

    def test_changed_pairs_rejected(self):
        self.run_pipeline()
        self.samples.reverse()
        self.save()
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_pipeline(resume=True)

    def test_nonempty_output_rejected(self):
        self.run_pipeline()
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            self.run_pipeline()

    def test_duplicate_and_unsafe_ids_rejected(self):
        for sid in ("second", "../escape", "/absolute"):
            self.samples[1]["sample_id"] = sid
            self.save()
            with self.assertRaisesRegex(ValueError, "sample_id"):
                self.run_pipeline()

    def test_missing_input_rejected_before_run(self):
        (self.root / "style.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_pipeline()
        self.assertFalse(self.out.exists())

    def test_typed_overrides(self):
        cfg = apply_overrides({}, ["save_images=false", "num_steps=4", "method=adain"])
        self.assertIs(cfg["save_images"], False)
        self.assertEqual(cfg["num_steps"], 4)
        self.assertEqual(cfg["method"], "adain")

    def test_empty_artifact_retried(self):
        self.run_pipeline()
        (self.out / "predictions/second/second.png").write_bytes(b"")
        self.run_pipeline(resume=True)
        self.assertEqual(len(self.records()), 3)

    def test_isolated_worker_pipeline_and_resume(self):
        self.cfg["worker_command"] = [sys.executable, "-u"]
        self.save()
        self.run_pipeline()
        self.assertEqual([r["status"] for r in self.records()], ["ok", "ok"])
        self.assertTrue((self.out / "logs/worker.log").is_file())
        self.run_pipeline(resume=True)
        self.assertEqual(len(self.records()), 2)

    def test_worker_reuses_process_after_sample_error(self):
        from backends.worker import WorkerBackend
        self.out.mkdir()
        cfg = {**self.cfg, "worker_command": [sys.executable, "-u"]}
        with __import__("contextlib").closing(WorkerBackend(cfg, Path(__file__).resolve().parents[1], self.out)) as worker:
            pid = worker.process.pid
            with self.assertRaisesRegex(RuntimeError, "Requested mock failure"):
                worker.infer({"sample_id": "bad", "mock_fail": True}, cfg, str(self.out))
            result = worker.infer({"sample_id": "good"}, cfg, str(self.out))
            self.assertTrue(Path(result.image_path).exists())
            self.assertEqual(pid, worker.process.pid)
            self.assertIsNone(worker.process.poll())
        self.assertIsNotNone(worker.process.returncode)

    def test_worker_command_is_not_shell_string(self):
        self.cfg["worker_command"] = "python -u"
        self.save()
        with self.assertRaisesRegex(ValueError, "argv list"):
            self.run_pipeline()

    def test_all_four_methods_in_isolated_worker(self):
        from evaluation.inputs import prepare_inputs
        for method in ("adain", "stytr2", "styleid", "scflow"):
            self.cfg.update(method=method, worker_command=[sys.executable, "-u"])
            self.out = self.root / method
            self.save()
            self.run_pipeline()
            self.assertEqual([r["status"] for r in self.records()], ["ok", "ok"])
            self.assertTrue(all(r["method"] == method and r["simulated"] for r in self.records()))
            report = prepare_inputs(self.out, allow_simulated=True)
            self.assertEqual(report["accepted_samples"], 2)
            self.assertTrue(report["complete"])


if __name__ == "__main__":
    unittest.main()
