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

    # Every surface form the precise cue pattern matches. Enumerated rather than
    # derived so a reader can see exactly what routes to "remove" -- the pattern
    # is built from stems ("remov" + "e|ing") and cannot be read off as literals.
    ALL_CUE_FORMS = [
        "remove", "removing", "delete", "deleting", "eliminate", "eliminating",
        "mute", "muting", "silence", "silencing", "reduce", "reducing",
        "suppress", "suppressing", "kill", "killing", "strip",
        "take out", "taking out", "took out",
        "take away", "taking away", "took away",
        "cut out", "cutting out",
        "strip out", "stripping out",
        "get rid of", "getting rid of", "got rid of",
        "filter out", "filtering out",
        "drown out", "drowning out",
        "clean out", "cleaning out",
        "without the", "minus the", "less of",
    ]

    # The -s/-ed forms of the same phrasal verbs, which are NOT precise cues.
    # They read as removals in isolation, but in a sentence they are the finite
    # verb of a passive or descriptive clause whose object is the sound to KEEP
    # ("her voice is drowned out by the music"). They belong to the loose tier:
    # never "remove", never "separate".
    DEMOTED_PHRASAL_FORMS = [
        "takes out", "takes away", "cuts out",
        "strips out", "stripped out", "gets rid of",
        "filters out", "filtered out",
        "drowns out", "drowned out",
        "cleans out", "cleaned out",
    ]

    def test_every_cue_form_is_strippable(self):
        """No cue may survive into the prompt -- that is the whole point."""
        for cue in self.ALL_CUE_FORMS:
            desc = f"{cue} the guitar"
            with self.subTest(cue=cue):
                self.assertEqual(clean_cli.strip_removal_cue(desc), "guitar")

    def test_every_cue_form_also_routes_to_remove(self):
        """Routing and stripping run the same regex; prove they agree.

        If a form strips cleanly but does not route, the prompt is never
        derived; if it routes but does not strip, SAM is handed the cue itself.
        """
        for cue in self.ALL_CUE_FORMS:
            desc = f"{cue} the guitar"
            with self.subTest(cue=cue):
                self.assertEqual(clean_cli.choose_method(desc), "remove")

    def test_demoted_phrasal_forms_denoise_even_beside_a_named_source(self):
        """A mixture word must not promote a loose-tier form back to "remove".

        "drowns out the guitar" names a source SAM can extract, which is exactly
        what makes the misroute dangerous: it would be handed the prompt
        "guitar" and the user would get back everything except the guitar.
        """
        for cue in self.DEMOTED_PHRASAL_FORMS:
            desc = f"{cue} the guitar"
            with self.subTest(cue=cue):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_a_phrasal_cue_beats_the_simple_cue_inside_it(self):
        """Alternation order is load-bearing: "strip out" must win over "strip".

        Matching only "strip" leaves the particle behind and sends SAM the
        prompt "out the piano".
        """
        self.assertEqual(clean_cli.strip_removal_cue("strip out the piano"), "piano")
        self.assertEqual(
            clean_cli.strip_removal_cue("stripping out the piano"), "piano")

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
