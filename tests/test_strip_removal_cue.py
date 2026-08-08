"""The SAM prompt for a removal job is the description minus the removal cue.

Sending the raw description would hand SAM "remove the barking dog" as the sound
to extract, which is the bug this whole route exists to fix.
"""
import unittest

import clean_cli


class StripRemovalCueTests(unittest.TestCase):
    def test_strips_cue_and_leading_article(self):
        cases = {
            "remove the barking dog": "barking dog",
            "get rid of the guitar": "guitar",
            "filter out the siren": "siren",
            "take out the traffic noise": "traffic noise",
            "cut out the keyboard typing": "keyboard typing",
            "suppress the crowd noise": "crowd noise",
            "eliminate the applause": "applause",
            "strip out the piano": "piano",
            "delete the dog barking": "dog barking",
            "mute the typing": "typing",
        }
        for desc, expected in cases.items():
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.strip_removal_cue(desc), expected)

    def test_every_cue_is_strippable(self):
        """No cue may survive into the prompt -- that is the whole point."""
        for cue in clean_cli._REMOVAL_CUES:
            desc = f"{cue} the guitar"
            with self.subTest(cue=cue):
                self.assertEqual(clean_cli.strip_removal_cue(desc), "guitar")

    def test_mid_sentence_cue_leaves_the_source_named(self):
        """Awkward but accepted: the remainder is not re-flowed as English.

        The prompt is clumsy; what matters is that the mixture word survives so
        SAM still has the source to extract. Documented in the spec's non-goals.
        """
        out = clean_cli.strip_removal_cue(
            "there is an alarm going off in the background, please remove it")
        self.assertIn("alarm", out)
        self.assertNotIn("remove", out)

    def test_empty_and_cue_only_return_empty(self):
        """An empty prompt vetoes the route (Task 5) rather than reaching SAM."""
        for desc in ["", "   ", "remove", "get rid of", "remove the"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.strip_removal_cue(desc), "")


if __name__ == "__main__":
    unittest.main()
