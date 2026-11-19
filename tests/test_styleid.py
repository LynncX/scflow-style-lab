"""Adapter tests with fake model/CUDA objects, not a real StyleID smoke test."""
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backends.mock import mock_png
from backends.styleid import StyleIDBackend, make_options


class StyleIDTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="styleid adapter ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "clip").mkdir()
        self.cfg = {"image_size": 512, "styleid": {"source": "upstream", "model_config": "model.yaml", "ckpt": "model.ckpt", "text_encoder": "clip"}}
        self.sample = {"sample_id": "pair_1", "content_path": str(self.root / "content a.v1.png"), "style_path": str(self.root / "style b.v2.png")}

    def test_exact_pairs_upstream_defaults_no_cache(self):
        options = make_options(self.root, self.cfg, self.sample, self.root)
        self.assertEqual((options.content_file, options.style_file), ("content a.v1.png", "style b.v2.png"))
        self.assertEqual(options.seed, 22)
        self.assertEqual((options.ddim_inv_steps, options.save_feat_steps, options.start_step), (50, 50, 49))
        self.assertEqual(options.precomputed, "")
        self.cfg["seed"] = 123
        self.assertEqual(make_options(self.root, self.cfg, self.sample, self.root).seed, 123)

    def test_cache_and_wrong_size_rejected(self):
        self.cfg["styleid"]["cache"] = True
        with self.assertRaisesRegex(ValueError, "cache"):
            make_options(self.root, self.cfg, self.sample, self.root)
        self.cfg["styleid"]["cache"] = False
        self.cfg["image_size"] = 256
        with self.assertRaisesRegex(ValueError, "512"):
            make_options(self.root, self.cfg, self.sample, self.root)

    def test_model_reused_and_actual_unet_calls_counted(self):
        unet = SimpleNamespace()
        hook = Mock()
        def register(callback):
            unet.callback = callback
            return hook
        unet.register_forward_pre_hook = register
        model = SimpleNamespace(model=SimpleNamespace(diffusion_model=unet))
        model_config = SimpleNamespace(model=SimpleNamespace(params=SimpleNamespace(cond_stage_config=SimpleNamespace(params={}))))
        upstream = SimpleNamespace(seed_everything=Mock(), OmegaConf=SimpleNamespace(load=Mock(return_value=model_config)),
                                   load_model_from_config=Mock(return_value=model))
        def infer(options, runtime):
            self.assertIs(runtime["model"], model)
            for _ in range(7):
                unet.callback(unet, ())
            (Path(options.output_path) / f"{options.sample_id}.png").write_bytes(mock_png(2))
        upstream.main = Mock(side_effect=infer)
        cuda = SimpleNamespace(is_available=lambda: True, set_device=Mock(), current_device=lambda: 0,
                               synchronize=Mock(), reset_peak_memory_stats=Mock(), max_memory_allocated=lambda: 1024)
        torch = SimpleNamespace(cuda=cuda, device=lambda value: SimpleNamespace(type="cuda", index=None),
                                no_grad=nullcontext, __version__="fake-not-real-torch")
        backend = StyleIDBackend(self.root)
        with patch.dict(sys.modules, {"torch": torch}), patch.object(sys, "path", sys.path.copy()), \
             patch("backends.styleid.validate_source", return_value={"commit": "fake"}), \
             patch("backends.styleid.importlib.import_module", return_value=upstream):
            first = backend.infer(self.sample, self.cfg, str(self.root))
            second = backend.infer(self.sample, self.cfg, str(self.root))
        self.assertEqual(upstream.load_model_from_config.call_count, 1)
        self.assertEqual(upstream.main.call_count, 2)
        self.assertEqual((first.nfe, second.nfe), (7, 7))
        self.assertEqual(second.timing_ms["model_load_ms"], 0)
        self.assertEqual(hook.remove.call_count, 2)
        self.assertEqual(model_config.model.params.cond_stage_config.params["version"], str(self.root / "clip"))

    def test_no_cuda_fails_explicitly(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            StyleIDBackend(self.root).infer(self.sample, self.cfg, self.root)

    def test_unsupported_steps_not_silently_ignored(self):
        self.cfg["styleid"]["ddim_inv_steps"] = 20
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            make_options(self.root, self.cfg, self.sample, self.root)

    def test_new_algorithms_cannot_import_into_controller(self):
        from runner import make_backend
        for method in ("adain", "stytr2", "styleid"):
            with self.assertRaisesRegex(ValueError, "worker_command"):
                make_backend({"backend": method}, self.root)
