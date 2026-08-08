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


class MixtureWordsMatchOnWordBoundariesTests(unittest.TestCase):
    """A mixture word buried inside a longer word is not evidence of a mixture.

    Substring matching read "engine" out of "engineer", "bass" out of "embassy"
    and "music" out of "musician", so ordinary cleanup requests were routed to
    SAM. Every case below was measured misrouting to "separate" before word
    boundaries were introduced.
    """

    def test_embedded_words_do_not_trigger_separation(self):
        for desc in [
            "an interview with an engineer",
            "clean up this recording from the embassy",
            "a musician talking about her career",
            "a crowded room, one speaker",
            "the speaker sounds alarmed",
            "clean up this dogged rambling monologue",
            "embark on the interview",
            "the professor discusses birdsong",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_plurals_still_count_as_a_mixture(self):
        """Word boundaries must not lose the plural forms."""
        for desc in ["two guitars playing", "the drums", "birds chirping",
                     "engines revving"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")

    def test_whole_words_still_separate(self):
        """The plan's mandated separation cases are unaffected."""
        for desc in ["the guitar", "a dog barking", "the voice over the music"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")


class RemovalPhrasingReturnsTheResidualTests(unittest.TestCase):
    """"Remove the X" runs SAM on X and hands back the RESIDUAL.

    This inverts the ruling in 40edf52, which sent all removal phrasing to
    denoise. That reasoning was about returning SAM's TARGET: when SAM finds
    nothing, the target is a near-empty stem and the voice is lost silently.
    It does not transfer to returning SAM's RESIDUAL -- when SAM finds nothing
    the residual is approximately the whole input, so the failure is a benign
    under-clean, the same reversible direction 40edf52 chose denoise for.

    All 13 descriptions below were must-denoise assertions under the old
    ruling. They are retained here as must-remove so the boundary stays pinned
    and the corpus does not shrink.
    """

    def test_removal_of_a_named_source_routes_to_remove(self):
        for desc in [
            "remove the barking dog",
            "take out the traffic noise",
            "cut out the keyboard typing",
            "remove the engine noise from this recording",
            "suppress the crowd noise",
            "there is an alarm going off in the background, please remove it",
            "get rid of the music",
            "filter out the siren",
            "eliminate the applause",
            "strip out the piano",
            "mute the typing",
            "delete the dog barking",
            "remove the tv in the background",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "remove")

    def test_removal_without_a_named_source_still_denoises(self):
        """The boundary that must NOT move: hiss and noise are not sources.

        Asking SAM to extract "hiss" is the single-speaker case that returns an
        empty stem. These stay on the denoiser.
        """
        for desc in [
            "remove the background noise",
            "remove the noise in the background",
            "remove the hiss",
            "get rid of the static",
            "remove noise from this phone message",
            "clean this up and remove the hum",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_explicit_method_still_wins(self):
        self.assertEqual(
            clean_cli.choose_method("remove the barking dog", "denoise"), "denoise")
        self.assertEqual(
            clean_cli.choose_method("clean this up", "remove"), "remove")

    def test_naming_a_source_to_keep_still_separates(self):
        """The supported way to extract a source is to NAME it, not remove it."""
        for desc in ["the guitar", "a dog barking", "the voice over the music"]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")


class RemovalCuesMatchOnWordBoundariesTests(unittest.TestCase):
    """A removal cue buried inside a longer word is not a removal request.

    Bare substring matching read "kill" out of "killer", "mute" out of
    "commuter", "strip" out of "stripped-back" and "less of" out of "unless of".
    On this route that is not a misroute but an INVERSION: the description is cut
    at the false match, so "the killer bass line" reached SAM as "er bass line"
    and the user was handed everything EXCEPT the sound they named, silently,
    reported as success. Every case below was measured routing to "remove".
    """

    def test_embedded_cues_do_not_route_to_remove(self):
        for desc, expected in [
            # No removal intent at all -- a named source, so extract it.
            ("the killer bass line", "separate"),
            ("a skilled musician playing guitar", "separate"),
            ("commuter traffic in the background", "separate"),
            ("unless of course the guitar is too loud", "separate"),
            # A real but ambiguous inflection: denoise, never separate.
            ("a guitar with a stripped-back drum track", "denoise"),
            ("the piano is muted", "denoise"),
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), expected)

    def test_no_embedded_cue_is_cut_out_of_the_prompt(self):
        """Routing and prompt derivation must agree on what the cue was.

        Both now run the same regex, so a description with no real cue survives
        the strip intact instead of losing a fragment of a longer word.
        """
        for desc, expected in [
            ("the killer bass line", "killer bass line"),
            ("a skilled musician playing guitar", "skilled musician playing guitar"),
            ("commuter traffic in the background", "commuter traffic in the background"),
            ("a guitar with a stripped-back drum track",
             "guitar with a stripped-back drum track"),
            ("the piano is muted", "piano is muted"),
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.strip_removal_cue(desc), expected)


class RemovalGerundsStillRouteToRemoveTests(unittest.TestCase):
    """"removing the guitar" asks for a removal, not for the guitar.

    These matched NO cue under bare-substring matching, because an e-final verb
    drops the e in its gerund -- "remove" + "ing" is not "removing". They fell
    through to the mixture check and routed to `separate`, which returns the
    named sound ALONE: everything the user wanted to keep, discarded. That is
    this project's founding failure, reached through ordinary English.
    """

    def test_gerund_and_inflected_forms_route_to_remove(self):
        cases = {
            # e-final stems: the gerund drops the e.
            "removing the guitar": "guitar",
            "muting the typing": "typing",
            "deleting the dog barking": "dog barking",
            "eliminating the applause": "applause",
            "reducing the traffic noise": "traffic noise",
            "silencing the alarm": "alarm",
            # Plain stems: the suffix just appends.
            "suppressing the crowd noise": "crowd noise",
            "killing the engine noise": "engine noise",
            # Phrasal cues inflect on the FIRST word, never the whole phrase.
            "taking out the traffic noise": "traffic noise",
            "took out the drums": "drums",
            "getting rid of the music": "music",
            "cutting out the keyboard typing": "keyboard typing",
            "filtering out the siren": "siren",
            "drowning out the applause": "applause",
            "cleaning out the crowd noise": "crowd noise",
            "stripping out the piano": "piano",
            "taking away the drums": "drums",
        }
        for desc, prompt in cases.items():
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "remove")
                self.assertEqual(clean_cli.strip_removal_cue(desc), prompt)


class AmbiguousRemovalPhrasingNeverSeparatesTests(unittest.TestCase):
    """Phrasing that MIGHT be a removal must never reach `separate`.

    This is the trap in tightening the cue pattern. `separate` returns the named
    sound alone -- the exact inverse of a removal request -- so a genuine removal
    phrasing the precise pattern misses would fall through to the mixture check
    and produce the worst possible output. The loose tier catches those forms and
    forces denoise instead: it under-cleans, which is recoverable, and it is the
    same reversible direction the whole routing rule is built on.

    The forms below are ambiguous by construction. "stripping the guitar" is a
    removal; "a stripped-back drum track" is a description of a mix; no regex can
    tell them apart. Declining to derive a prompt from them is correct.
    """

    def test_ambiguous_inflections_denoise_rather_than_separate(self):
        for desc in [
            "stripping the guitar",
            "a guitar with a stripped-back drum track",
            "she muted the guitar",
            "the piano is muted",
            "the traffic noise was reduced",
            "the applause was eliminated",
            "he kills the drums in the second verse",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")


class BackgroundDeviceIsAMixtureTests(unittest.TestCase):
    """A device word counts only alongside evidence it is audibly playing.

    Dropping "radio"/"phone"/"tv"/"television" outright to stop them naming the
    recording medium also lost a real class of mixture -- "a tv playing in the
    background" -- because _MIXTURE_PHRASES does not cover "in the background".
    Requiring a playing cue restores those without reopening the misroute.
    """

    def test_device_plus_playing_cue_separates(self):
        for desc in [
            "there is a tv playing in the background while she talks",
            "someone is talking on the phone in the background",
            "a radio is playing in the background of this call",
            "the television is on in the background",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "separate")

    def test_device_without_a_cue_still_denoises(self):
        """The recording-medium reading must stay the default."""
        for desc in [
            "clean up this phone call recording",
            "improve this radio interview",
            "clean up this tv recording",
            "clean up the audio from this television segment",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")

    def test_background_noise_is_not_a_second_source(self):
        """Hiss and hum in "the background" are noise, not a device."""
        for desc in [
            "remove the background noise",
            "remove the noise in the background",
            "there is a hiss in the background",
            "reduce the background hum",
        ]:
            with self.subTest(desc=desc):
                self.assertEqual(clean_cli.choose_method(desc), "denoise")


class RemoveDispatchTests(unittest.TestCase):
    """An unusable prompt must veto the route rather than reach SAM.

    handle() defaults an empty description to "speech" -- on a removal job that
    would extract exactly the thing the user wants to keep, then hand back the
    residual, deleting the voice. The veto is what prevents that.
    """

    def test_empty_prompt_downgrades_to_denoise(self):
        # "remove the" has a cue and no source, so routing already says denoise;
        # the veto covers the case where a source word survives routing but the
        # prompt strips to nothing.
        self.assertEqual(clean_cli.strip_removal_cue("remove the"), "")
        self.assertEqual(clean_cli.choose_method("remove the"), "denoise")

    def test_explicit_remove_with_unusable_prompt_is_vetoed_too(self):
        """An explicit method key does not make a blank prompt usable."""
        self.assertEqual(clean_cli.choose_method("remove", "remove"), "remove")
        self.assertEqual(clean_cli.strip_removal_cue("remove"), "")


if __name__ == "__main__":
    unittest.main()
