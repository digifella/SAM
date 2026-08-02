from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import run_sam_interactive as r


def _make_audio_dirs():
    """Create a minimal real (tiny) wav plus an output dir, for a real call
    into process_audio_file (not a mock). model/processor are left as None:
    the cancellation checkpoint fires before either is touched, and the
    genuine-error test relies on the None processor blowing up naturally."""
    td = tempfile.mkdtemp()
    audio_path = Path(td) / "in.wav"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    output_dir = Path(td) / "out"
    output_dir.mkdir()
    return audio_path, output_dir


class ProcessAudioFileCancellationPropagationTests(unittest.TestCase):
    """F2b: process_audio_file's own catch-all except must not re-wrap a
    JobCancelledError into a plain RuntimeError. worker/handlers/sam_audio_cleanup.py
    always calls this with raise_on_error=True, so this is the exact condition
    the worker hits on every cancellation inside this function."""

    def test_cancellation_propagates_as_job_cancelled_error(self):
        audio_path, output_dir = _make_audio_dirs()

        with self.assertRaises(r.JobCancelledError) as ctx:
            r.process_audio_file(
                audio_path=audio_path,
                description="x",
                output_dir=output_dir,
                model=None,
                processor=None,
                device="cpu",
                memory_fraction=0.9,
                rerank=1,
                predict_spans=False,
                chunk_duration=30.0,
                overlap=1.0,
                convert_to_mono=False,
                progress_cb=None,
                is_cancelled_cb=lambda: True,
                raise_on_error=True,
            )

        # Not just isinstance-compatible: the exact type must be
        # JobCancelledError, not a plain RuntimeError wrapper around it.
        self.assertIs(type(ctx.exception), r.JobCancelledError)
        # The message must be the original, unwrapped message (no "(log: ...)"
        # suffix from the generic error-wrap path). is_cancelled_cb=True fires
        # at the very first checkpoint in the function.
        self.assertEqual(str(ctx.exception), "Cancelled before preprocessing")

    def test_genuine_error_is_still_wrapped_as_runtime_error_with_log_path(self):
        audio_path, output_dir = _make_audio_dirs()

        with self.assertRaises(RuntimeError) as ctx:
            r.process_audio_file(
                audio_path=audio_path,
                description="x",
                output_dir=output_dir,
                model=None,
                processor=None,  # calling processor(...) raises TypeError
                device="cpu",
                memory_fraction=0.9,
                rerank=1,
                predict_spans=False,
                chunk_duration=30.0,
                overlap=1.0,
                convert_to_mono=False,
                progress_cb=None,
                is_cancelled_cb=lambda: False,
                raise_on_error=True,
            )

        exc = ctx.exception
        # A genuine (non-cancellation) error must still be wrapped exactly as
        # before: plain RuntimeError, not JobCancelledError, log path present,
        # original exception chained via __cause__.
        self.assertIs(type(exc), RuntimeError)
        self.assertNotIsInstance(exc, r.JobCancelledError)
        self.assertIn("(log:", str(exc))
        self.assertIsInstance(exc.__cause__, TypeError)

    def test_finally_cleanup_still_runs_on_cancellation(self):
        # convert_to_mono=True makes process_audio_file create a real temp
        # file (via ffmpeg) and register it in temp_files before the second
        # cancellation checkpoint fires. The finally: block must still unlink
        # it even though the try block exits via a raised JobCancelledError.
        audio_path, output_dir = _make_audio_dirs()

        calls = {"n": 0}

        def cancel_after_conversion():
            calls["n"] += 1
            # False on the first check (before preprocessing/conversion),
            # True on the second (before inference) -- so conversion runs
            # and creates a temp file, then cancellation fires afterward.
            return calls["n"] >= 2

        before = set(Path(tempfile.gettempdir()).glob("sam_mono16k_*"))

        with self.assertRaises(r.JobCancelledError):
            r.process_audio_file(
                audio_path=audio_path,
                description="x",
                output_dir=output_dir,
                model=None,
                processor=None,
                device="cpu",
                memory_fraction=0.9,
                rerank=1,
                predict_spans=False,
                chunk_duration=30.0,
                overlap=1.0,
                convert_to_mono=True,
                progress_cb=None,
                is_cancelled_cb=cancel_after_conversion,
                raise_on_error=True,
            )

        self.assertGreaterEqual(calls["n"], 2)
        after = set(Path(tempfile.gettempdir()).glob("sam_mono16k_*"))
        self.assertEqual(after - before, set())


if __name__ == "__main__":
    unittest.main()
