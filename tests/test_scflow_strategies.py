"""Exercise SCFlow control logic without checkpoints, torch or CUDA."""
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from backends.scflow import SCFlowBackend, fuse_styles, integrate
from experiment_inputs import integration_options, style_references


class FakeFlow:
    val_step_num = 9

    def __init__(self):
        self.model = Mock()
        self.calls = []
        self.fail_on = None
        self.hook = self.model.register_forward_hook.return_value

    def predict_forward(self, source):
        self.calls.append((self.val_step_num, source))
        if self.val_step_num == self.fail_on:
            raise RuntimeError("solver failed")
        callback = self.model.register_forward_hook.call_args.args[0]
        # Deliberately differ from the requested step count to catch fake NFE.
        for _ in range(self.val_step_num + 1):
            callback(self.model, (), None)
        return f"prediction-{self.val_step_num}"


class StrategyTests(unittest.TestCase):
    def test_adaptive_stay_boundary_and_fallback(self):
        source = object()
        for gap, selected, expected_nfe in ((0.2, 1, 2), (0.5, 1, 2), (0.6, 4, 7)):
            with self.subTest(gap=gap), patch("backends.scflow.endpoint_gap", return_value=gap):
                module = FakeFlow()
                pred, nfe, info = integrate(module, source, 1, 0.5)
                self.assertEqual(pred, f"prediction-{selected}")
                self.assertEqual(nfe, expected_nfe)
                self.assertEqual(info["fallback"], selected == 4)
                self.assertEqual(info["selected_steps"], selected)
                self.assertEqual(info["one_step_endpoint_rmse"], gap)
                self.assertEqual(module.calls, [(1, source)] + ([(4, source)] if selected == 4 else []))
                module.hook.remove.assert_called_once()
                self.assertEqual(module.val_step_num, 9)

    def test_fixed_does_not_compute_gap(self):
        with patch("backends.scflow.endpoint_gap") as gap:
            pred, nfe, info = integrate(FakeFlow(), object(), 4, None)
        self.assertEqual((pred, nfe, info["selected_steps"]), ("prediction-4", 5, 4))
        gap.assert_not_called()

    def test_failure_restores_solver_and_hook(self):
        module = FakeFlow()
        module.fail_on = 4
        with patch("backends.scflow.endpoint_gap", return_value=1), self.assertRaisesRegex(RuntimeError, "solver"):
            integrate(module, object(), 1, 0.5)
        self.assertEqual(module.val_step_num, 9)
        module.hook.remove.assert_called_once()

    def test_backend_combines_fusion_and_fallback_then_decodes_once(self):
        module = FakeFlow()
        module.dtype = "float32"
        predictions = {1: MagicMock(), 4: MagicMock()}
        real_predict = module.predict_forward

        def predict(source):
            real_predict(source)
            return predictions[module.val_step_num]

        module.predict_forward = predict
        one_a, one_b = MagicMock(), MagicMock()
        predictions[1].__getitem__.side_effect = [one_a, one_b]
        difference = one_a.float.return_value.__sub__.return_value
        difference.square.return_value.mean.return_value.sqrt.return_value.item.return_value = 0.75
        module.get_sample_to_vis = Mock()
        torch = Mock()
        torch.cuda.is_available.return_value = False
        torch.device.return_value = SimpleNamespace(type="cpu")
        torch.no_grad.side_effect = nullcontext

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("flow", "decoder"):
                (root / name).touch()
            backend = SCFlowBackend(root)
            backend._loaded = ((root / "third_party/scflow_pkg/inference.yaml",
                                root / "flow", root / "decoder", 1, True, None), module)
            cfg = {"scflow": {"ckpt": "flow", "unclip_ckpt": "decoder",
                              "integration": "adaptive", "adaptive_threshold": 0.5, "fusion": "weighted"}}
            with patch.dict("sys.modules", {"torch": torch,
                                            "omegaconf": SimpleNamespace(OmegaConf=Mock()),
                                            "scflow.trainer_module": SimpleNamespace(TrainerModuleSCFlow=Mock())}), \
                 patch("backends.scflow.CLIPEncoder") as encoder, \
                 patch("backends.scflow.fuse_styles", wraps=fuse_styles) as fuse:
                feature = MagicMock()
                encoder.return_value.encode_image.return_value = feature
                sample = {"sample_id": "test", "content_path": "c", "style_paths": ["a", "b"], "style_weights": [3, 1]}
                result = backend.infer(sample, cfg, directory)
            self.assertEqual([c.args[0] for c in encoder.return_value.encode_image.call_args_list],
                             [root / "c", root / "a", root / "b"])
            self.assertEqual(fuse.call_args.args[:2], (["a", "b"], [0.75, 0.25]))
            self.assertEqual(result.nfe, 7)
            self.assertEqual(result.provenance["selected_steps"], 4)
            module.get_sample_to_vis.assert_called_once()
            branch = predictions[4].__getitem__.return_value
            self.assertIs(module.get_sample_to_vis.call_args.args[0],
                          branch.__add__.return_value.__truediv__.return_value)

    def test_invalid_integration_options(self):
        for options in ({}, {"adaptive_threshold": -1}, {"adaptive_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                integration_options({"scflow": {"integration": "adaptive", **options}})
        with self.assertRaises(ValueError):
            integration_options({"num_steps": 4, "scflow": {"integration": "adaptive", "adaptive_threshold": 0.5}})

    def test_fuses_encoded_features_and_preserves_single_reference(self):
        class Feature(float):
            def float(self):
                return float(self)

        encode = Mock(side_effect={"a": Feature(2), "b": Feature(10)}.__getitem__)
        sample = {"style_paths": ["a", "b"], "style_weights": [3, 1]}
        for mode, expected in (("mean", 6), ("weighted", 4)):
            encode.reset_mock()
            self.assertEqual(fuse_styles(*style_references(sample, {"scflow": {"fusion": mode}}), encode), expected)
            self.assertEqual([c.args[0] for c in encode.call_args_list], ["a", "b"])
        self.assertIsInstance(fuse_styles(*style_references({"style_path": "a"}, {}), encode), Feature)

    def test_rejects_ambiguous_and_bad_reference_weights(self):
        cfg = {"scflow": {"fusion": "weighted"}}
        for weights in (None, [], [1], [0, 0], [1, -1], [1, float("inf")], [True, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                style_references({"style_paths": ["a", "b"], "style_weights": weights}, cfg)
        with self.assertRaises(ValueError):
            style_references({"style_path": "a", "style_paths": ["b"]}, {})
