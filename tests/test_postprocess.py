from __future__ import annotations

import unittest

from worker.handlers.sam_audio_cleanup import _parse_loudnorm_json

SAMPLE_STDERR = """\
ffmpeg version 6.1.1 Copyright (c) 2000-2023
  Stream #0:0: Audio: pcm_s16le, 48000 Hz, mono, s16
[Parsed_loudnorm_0 @ 0x5b7e8c]
{
    "input_i" : "-27.61",
    "input_tp" : "-4.47",
    "input_lra" : "6.30",
    "input_thresh" : "-38.21",
    "output_i" : "-16.58",
    "output_tp" : "-3.00",
    "output_lra" : "5.90",
    "output_thresh" : "-27.07",
    "normalization_type" : "dynamic",
    "target_offset" : "0.58"
}
"""


class ParseLoudnormJsonTests(unittest.TestCase):
    def test_parses_measured_values(self):
        m = _parse_loudnorm_json(SAMPLE_STDERR)
        self.assertEqual(m["input_i"], "-27.61")
        self.assertEqual(m["input_tp"], "-4.47")
        self.assertEqual(m["input_lra"], "6.30")
        self.assertEqual(m["input_thresh"], "-38.21")
        self.assertEqual(m["target_offset"], "0.58")

    def test_no_json_block_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json("ffmpeg version 6.1.1\nno json here\n")

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('prefix\n{\n    "input_i" : "-27.61",\n')

    def test_missing_key_raises(self):
        with self.assertRaises(ValueError):
            _parse_loudnorm_json('{\n  "input_i" : "-27.61"\n}\n')


if __name__ == "__main__":
    unittest.main()
