from __future__ import annotations

import unittest

import clean_cli


class ChooseMethodTests(unittest.TestCase):
    def test_cleanup_language_routes_to_denoise(self):
        for desc in ["clean this up", "remove the background noise",
                     "improve poor audio", "a person speaking", "speech", ""]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_named_source_in_a_mixture_routes_to_separate(self):
        for desc in ["the guitar", "a man speaking over a radio",
                     "the voice over the music", "a dog barking"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")

    def test_explicit_method_always_wins(self):
        self.assertEqual(clean_cli.choose_method("the guitar", "denoise"), "denoise")
        self.assertEqual(clean_cli.choose_method("clean this up", "separate"), "separate")

    def test_unknown_explicit_method_rejected(self):
        with self.assertRaises(ValueError):
            clean_cli.choose_method("anything", "magic")


if __name__ == "__main__":
    unittest.main()


class RecordingMediumIsNotAMixtureTests(unittest.TestCase):
    """Words naming the RECORDING MEDIUM must not force SAM separation.

    "radio", "phone", "tv" and "television" name how something was recorded at
    least as often as they name a competing sound. Treating them as mixture
    evidence sent plain cleanup requests -- "clean up this phone call recording",
    "improve this radio interview" -- to SAM, which on single-speaker noisy audio
    returns a target with almost no speech in it. That is the exact failure this
    routing exists to avoid, and the design rule is that ambiguity resolves to
    denoise.
    """

    def test_recording_medium_language_still_denoises(self):
        for desc in [
            "clean up this phone call recording",
            "improve this radio interview",
            "clean up this tv recording",
            "remove noise from this phone message",
            "clean up the audio from this television segment",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_a_real_mixture_still_separates_via_phrase(self):
        """Dropping those words must not weaken genuine mixture detection."""
        self.assertEqual(
            clean_cli.choose_method("a man speaking over a radio"), "separate")
        self.assertEqual(
            clean_cli.choose_method("her voice over the television"), "separate")
