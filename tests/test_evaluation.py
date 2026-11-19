"""Evaluation preparation contracts, with explicitly simulated images only."""
import json
import unittest
import subprocess
import sys
from pathlib import Path

import test_mock_pipeline as pipeline
from evaluation.inputs import export_inputs, prepare_inputs


class EvaluationTests(unittest.TestCase):
    setUp = pipeline.PipelineTests.setUp
    save = pipeline.PipelineTests.save
    run_pipeline = pipeline.PipelineTests.run_pipeline
    records = pipeline.PipelineTests.records

    def test_requires_explicit_simulated_flag(self):
        self.run_pipeline()
        with self.assertRaisesRegex(ValueError, "allow-simulated"):
            prepare_inputs(self.out)

    def test_ordered_pairs_export_without_metrics(self):
        self.run_pipeline()
        path = export_inputs(self.out, allow_simulated=True)
        report = json.loads(path.read_text())
        self.assertEqual([p["sample_id"] for p in report["pairs"]], ["second", "first"])
        self.assertTrue(report["simulated"] and report["complete"])
        self.assertEqual(report["kind"], "evaluation_inputs_only")
        self.assertFalse((self.out / "metrics.json").exists())

    def test_failure_policy_is_explicit(self):
        self.samples[0]["mock_fail"] = True
        self.save()
        self.run_pipeline()
        with self.assertRaisesRegex(ValueError, "failed/missing"):
            prepare_inputs(self.out, allow_simulated=True)
        report = prepare_inputs(self.out, allow_simulated=True, allow_failures=True)
        self.assertFalse(report["complete"])
        self.assertEqual((report["total_samples"], report["accepted_samples"]), (2, 1))
        self.assertEqual(report["excluded"][0]["sample_id"], "second")

    def test_input_content_change_rejected(self):
        self.run_pipeline()
        (self.root / "style.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_bad_prediction_rejected(self):
        self.run_pipeline()
        (self.out / "predictions/second/second.png").write_bytes(b"not a png" * 10)
        with self.assertRaisesRegex(ValueError, "PNG"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_wrong_size_rejected(self):
        self.run_pipeline()
        (self.out / "predictions/second/second.png").write_bytes(pipeline.mock_png(8))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_tampered_snapshot_rejected(self):
        self.run_pipeline()
        path = self.out / "config.json"
        snapshot = json.loads(path.read_text())
        snapshot["config"]["image_size"] = 999
        path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_method_mismatch_rejected(self):
        self.run_pipeline()
        records = self.records()
        records[0]["method"] = "styleid"
        (self.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_feature_only_run_rejected(self):
        self.cfg.update(method="scflow", save_images=False)
        self.save()
        self.run_pipeline()
        with self.assertRaisesRegex(ValueError, "requires PNG"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_prediction_outside_run_rejected(self):
        self.run_pipeline()
        records = self.records()
        records[0]["image_path"] = str(self.root / "style.png")
        (self.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "outside"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_unknown_record_id_rejected(self):
        self.run_pipeline()
        records = self.records()
        records[0]["sample_id"] = "not_in_manifest"
        (self.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            prepare_inputs(self.out, allow_simulated=True)

    def test_cli_explicit_simulation_and_exit_code(self):
        self.run_pipeline()
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
        rejected = subprocess.run([sys.executable, str(script), str(self.out)], capture_output=True, text=True)
        self.assertEqual(rejected.returncode, 1)
        accepted = subprocess.run([sys.executable, str(script), str(self.out), "--allow-simulated"], capture_output=True, text=True)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue((self.out / "evaluation/inputs.json").exists())
