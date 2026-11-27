import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backends.mock import MockBackend, mock_png
from evaluation.inputs import prepare_inputs
from runner import run


class ReferenceManifestTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        for name in ("content", "ref1", "ref2", "ref3"):
            (self.root / f"{name}.png").write_bytes(mock_png(2))
        self.sample = {
            "sample_id": "example",
            "content_path": "content.png",
            "style_path": "ref1.png",
            "style_paths": ["ref1.png", "ref2.png", "ref3.png"],
        }
        self.manifest = self.root / "pairs.jsonl"
        self.config = self.root / "config.json"
        self.output = self.root / "output"

    def run_fixture(self, mode, resume=False):
        self.manifest.write_text(json.dumps(self.sample) + "\n", encoding="utf-8")
        self.config.write_text(json.dumps({
            "run_name": "test",
            "backend": "scflow",
            "manifest": str(self.manifest),
            "data_root": str(self.root),
            "image_size": 512,
            "scflow": {"style_fusion": mode},
        }), encoding="utf-8")
        with patch("runner.make_backend", return_value=MockBackend()):
            return run(self.config, out_dir_override=str(self.output), resume=resume)

    def test_all_selected_references_are_resolved_and_fingerprinted(self):
        self.run_fixture("weighted")
        snapshot = json.loads((self.output / "config.json").read_text(encoding="utf-8"))
        paths = [str(self.root / name) for name in self.sample["style_paths"]]
        self.assertEqual(snapshot["samples"][0]["style_paths"], paths)
        self.assertTrue(set(paths) <= set(snapshot["inputs_sha256"]))
        self.assertEqual(prepare_inputs(self.output)["pairs"][0]["style_paths"], paths)
        (self.root / "ref2.png").write_bytes(mock_png(3))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_fixture("weighted", resume=True)
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_inputs(self.output)

    def test_first_reference_must_match_single_reference(self):
        self.sample["style_paths"].reverse()
        with self.assertRaisesRegex(ValueError, "first"):
            self.run_fixture("mean")

    def test_single_only_reads_primary_reference(self):
        (self.root / "ref2.png").unlink()
        self.run_fixture("single")
        report = prepare_inputs(self.output)
        self.assertNotIn("style_paths", report["pairs"][0])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires PyTorch")
class FeatureFusionTests(unittest.TestCase):
    def test_mean_and_weighted_preserve_reference_order_and_raw_scale(self):
        import torch
        from backends.style_fusion import fuse_style_features

        references = torch.tensor([[2., 0.], [0., 2.], [4., 0.]])
        mean, uniform = fuse_style_features(references, "mean")
        torch.testing.assert_close(mean, references.mean(dim=0, keepdim=True))
        torch.testing.assert_close(uniform, torch.ones(3) / 3)

        fused, weights = fuse_style_features(references, "weighted", temperature=0.1)
        center = references.mean(dim=0, keepdim=True)
        similarities = torch.nn.functional.cosine_similarity(references, center, dim=1)
        expected_weights = torch.softmax(similarities / 0.1, dim=0)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(fused, (references * weights[:, None]).sum(dim=0, keepdim=True))
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertEqual(float(weights[0]), float(weights[2]))

    def test_rejects_nonfinite_features_and_invalid_temperature(self):
        import torch
        from backends.style_fusion import fuse_style_features

        with self.assertRaisesRegex(ValueError, "finite"):
            fuse_style_features(torch.full((3, 2), float("nan")), "weighted")
        with self.assertRaisesRegex(ValueError, "temperature"):
            fuse_style_features(torch.ones(3, 2), "weighted", temperature=0)


if __name__ == "__main__":
    unittest.main()
