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


if __name__ == "__main__":
    unittest.main()
