from __future__ import annotations

import unittest

import torch

from sam_audio_local.loader import (
    StubVisionEncoder,
    filter_vision_keys,
    strip_ranker_config,
    stubbed_vision_encoder,
)


class _FakeVisionCfg:
    dim = 1024


class StripRankerConfigTests(unittest.TestCase):
    def test_rankers_set_to_none_and_rest_preserved(self):
        config = {
            "in_channels": 768,
            "visual_ranker": {"kind": "imagebind"},
            "text_ranker": {"kind": "ensemble"},
            "span_predictor": "pe-a-frame-large",
        }
        out = strip_ranker_config(config)
        self.assertIsNone(out["visual_ranker"])
        self.assertIsNone(out["text_ranker"])
        self.assertEqual(out["in_channels"], 768)
        self.assertEqual(out["span_predictor"], "pe-a-frame-large")

    def test_input_dict_not_mutated(self):
        config = {"visual_ranker": {"kind": "imagebind"}, "text_ranker": None}
        strip_ranker_config(config)
        self.assertEqual(config["visual_ranker"], {"kind": "imagebind"})


class FilterVisionKeysTests(unittest.TestCase):
    def test_drops_only_vision_encoder_keys(self):
        sd = {
            "vision_encoder.blocks.0.weight": torch.zeros(1),
            "transformer.layers.0.weight": torch.zeros(1),
            "audio_codec.encoder.weight": torch.zeros(1),
        }
        out = filter_vision_keys(sd)
        self.assertNotIn("vision_encoder.blocks.0.weight", out)
        self.assertIn("transformer.layers.0.weight", out)
        self.assertIn("audio_codec.encoder.weight", out)


class StubVisionEncoderTests(unittest.TestCase):
    def test_has_dim_and_no_parameters(self):
        stub = StubVisionEncoder(_FakeVisionCfg())
        self.assertEqual(stub.dim, 1024)
        self.assertEqual(sum(p.numel() for p in stub.parameters()), 0)

    def test_forward_raises(self):
        stub = StubVisionEncoder(_FakeVisionCfg())
        with self.assertRaises(RuntimeError):
            stub([torch.zeros(1, 3, 8, 8)])


class StubbedVisionEncoderContextTests(unittest.TestCase):
    def test_patches_and_restores(self):
        from sam_audio.model import model as sam_model_module

        original = sam_model_module.PerceptionEncoder
        with stubbed_vision_encoder():
            self.assertIs(sam_model_module.PerceptionEncoder, StubVisionEncoder)
        self.assertIs(sam_model_module.PerceptionEncoder, original)

    def test_restores_on_exception(self):
        from sam_audio.model import model as sam_model_module

        original = sam_model_module.PerceptionEncoder
        with self.assertRaises(ValueError):
            with stubbed_vision_encoder():
                raise ValueError("boom")
        self.assertIs(sam_model_module.PerceptionEncoder, original)


from pathlib import Path

MODEL_DIR = Path.home() / "models" / "sam-audio-large-tv"


@unittest.skipUnless(
    (MODEL_DIR / "checkpoint.pt").exists(), "local model weights not present"
)
class LoadOptimizedIntegrationTests(unittest.TestCase):
    """Loads the real 14.9GB checkpoint -- takes a few minutes."""

    @classmethod
    def setUpClass(cls):
        from sam_audio_local.loader import load_sam_audio_optimized

        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.model = load_sam_audio_optimized(MODEL_DIR, device=cls.device)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_vision_encoder_is_stub_with_no_parameters(self):
        from sam_audio_local.loader import StubVisionEncoder

        self.assertIsInstance(self.model.vision_encoder, StubVisionEncoder)
        self.assertEqual(
            sum(p.numel() for p in self.model.vision_encoder.parameters()), 0
        )

    def test_rankers_are_none(self):
        self.assertIsNone(self.model.visual_ranker)
        self.assertIsNone(self.model.text_ranker)

    def test_span_predictor_present(self):
        self.assertTrue(hasattr(self.model, "span_predictor"))

    def test_dtypes(self):
        if self.device != "cuda":
            self.skipTest("fp16 cast only applies on CUDA")
        self.assertEqual(
            next(self.model.transformer.parameters()).dtype, torch.float16
        )
        self.assertEqual(
            next(self.model.audio_codec.parameters()).dtype, torch.float16
        )
        self.assertEqual(
            next(self.model.text_encoder.parameters()).dtype, torch.float32
        )

    def test_resident_vram_under_14gb(self):
        # Measured 12.84GB: fp16 DiT 5.9GB + fp32 span predictor 6.1GB
        # (pe-a-frame-large is 1.53B params, kept fp32 per design spec)
        # + fp32 T5 0.4GB + fp16 codec 0.2GB.
        if self.device != "cuda":
            self.skipTest("VRAM check requires CUDA")
        resident_gb = torch.cuda.memory_allocated() / 1e9
        self.assertLess(resident_gb, 14.0)


if __name__ == "__main__":
    unittest.main()
