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
