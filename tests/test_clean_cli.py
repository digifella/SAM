import fcntl
import io
import json
import shutil as _shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

import clean_cli


class CleanCliCoreTests(unittest.TestCase):
    def _run(self, td, extra_args=None, fake_handle=None, job_json_text=None):
        td = Path(td)
        inp = td / "in.wav"; inp.write_bytes(b"RIFFfake")
        jj = td / "job.json"
        # method is pinned to "separate" explicitly: these tests mock handle() to
        # exercise the SAM separation path specifically. clean_cli now routes by
        # default, and "speech" alone would send an unpinned payload to the denoise
        # path instead (explicit method always beats inference — see
        # clean_cli.choose_method). Do not remove this pin; it is what keeps these
        # tests testing the path they were written to test, not a "simplification".
        jj.write_text(json.dumps({"description": "speech", "method": "separate"})
                      if job_json_text is None else job_json_text)
        out = td / "out"
        # --gpu-lock-timeout kept short and GPU_LOCK_PATH patched to a
        # per-test tmp file below: this host runs clean_cli for real jobs, so
        # tests must never contend on the real ~/.sam_audio_gpu.lock (a held
        # real lock would otherwise hang the suite for up to the real
        # 7200s default). gpu_lock() itself and clean_cli.GPU_LOCK_PATH are
        # left untouched — a later task exercises the real flock directly.
        argv = ["--input", str(inp), "--job-json", str(jj), "--out-dir", str(out),
                "--gpu-lock-timeout", "5"]
        argv += (extra_args or [])

        def default_fake(input_path, input_data, job, progress_cb=None,
                         is_cancelled_cb=None, work_dir=None):
            progress_cb(50, "half way", "separate")
            zip_path = Path(work_dir) / "result_inner.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("target.wav", b"x"); zf.writestr("residual.wav", b"y")
            return {"output_data": {"description": "speech", "duration_seconds": 1.0},
                    "output_file": zip_path}

        with patch.object(clean_cli, "GPU_LOCK_PATH", td / "gpu_test.lock"), \
             patch.object(clean_cli, "handle", side_effect=fake_handle or default_fake):
            rc = clean_cli.main(argv)
        return rc, out

    def test_success_writes_done_status_and_result_zip(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td)
            self.assertEqual(rc, 0)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertTrue((out / "result.zip").exists())
            # zip_src.parent == out_dir here (default_fake writes straight into
            # work_dir) — proves the rmtree guard does NOT delete out_dir itself
            # (status.json above and result.zip both survive inside it).
            self.assertTrue(out.exists())

    def test_handler_error_writes_error_status_and_rc1(self):
        def boom(**kw):
            raise RuntimeError("separation exploded")
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, fake_handle=boom)
            self.assertEqual(rc, 1)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "error")
            self.assertIn("separation exploded", status["error"])

    def test_nested_handler_workdir_moved_and_removed(self):
        # Mirrors the real handle(): products land under
        # <work_dir>/sam_handler_<job_id>/, not directly in work_dir.
        def nested_fake(input_path, input_data, job, progress_cb=None,
                        is_cancelled_cb=None, work_dir=None):
            nested = Path(work_dir) / "sam_handler_0"
            nested.mkdir()
            zip_path = nested / "result_inner.zip"
            import zipfile
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("target.wav", b"x"); zf.writestr("residual.wav", b"y")
            return {"output_data": {"description": "speech", "duration_seconds": 1.0},
                    "output_file": zip_path}

        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, fake_handle=nested_fake)
            self.assertEqual(rc, 0)
            self.assertTrue((out / "result.zip").exists())
            self.assertFalse((out / "sam_handler_0").exists())
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")

    def test_malformed_job_json_still_writes_error_status_and_rc1(self):
        # A job.json parse failure must not bypass the status.json contract —
        # the bridge treats a missing/never-updated status.json as
        # perpetually "submitted", so this must land as state:"error", not
        # an uncaught exception with no status.json at all.
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, job_json_text="{not valid json")
            self.assertEqual(rc, 1)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "error")

    def test_opus_encode_failure_still_succeeds_with_warning(self):
        # ffmpeg missing/broken must degrade, not destroy an already-successful
        # separation: state stays "done", result.zip is kept, and the failure
        # surfaces as a warning instead of failing the job.
        with tempfile.TemporaryDirectory() as td:
            with patch.object(clean_cli, "encode_opus",
                              side_effect=RuntimeError("ffmpeg not found")):
                rc, out = self._run(td, extra_args=["--opus"])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "result.zip").exists())
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertIn("opus encode failed", status["metadata"]["warning"])
            self.assertIn("ffmpeg not found", status["metadata"]["warning"])

    def test_denoise_method_produces_done_status_and_result_zip(self):
        # The other CleanCliCoreTests all pin method="separate" to exercise handle().
        # This test covers the other branch clean_cli.main() can now take: a
        # "method": "denoise" payload should never reach handle() at all, and should
        # still land the same state:"done" / result.zip contract. Needs a real
        # (short, synthetic) wav — denoise_file reads actual audio — but stays fast:
        # 7.5s at 8kHz, a quiet noise region followed by a louder "speech" region so
        # estimate_noise_profile has real headroom to find, no repo fixture involved.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            sr = 8000
            rng = np.random.default_rng(0)
            quiet = rng.uniform(-0.01, 0.01, int(3.5 * sr))
            loud = rng.uniform(-0.2, 0.2, int(4.0 * sr))
            audio = np.concatenate([quiet, loud]).astype(np.float32)
            inp = td / "in.wav"
            sf.write(inp, audio, sr)
            jj = td / "job.json"
            jj.write_text(json.dumps({"description": "clean this up", "method": "denoise"}))
            out = td / "out"
            argv = ["--input", str(inp), "--job-json", str(jj), "--out-dir", str(out),
                    "--gpu-lock-timeout", "5"]
            with patch.object(clean_cli, "GPU_LOCK_PATH", td / "gpu_test.lock"), \
                 patch.object(clean_cli, "handle") as mock_handle:
                rc = clean_cli.main(argv)
            self.assertEqual(rc, 0)
            mock_handle.assert_not_called()
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertEqual(status["metadata"]["method"], "spectral_subtraction")
            import zipfile
            with zipfile.ZipFile(out / "result.zip") as zf:
                names = set(zf.namelist())
            self.assertEqual(names, {"target.wav", "residual.wav", "metadata.json"})

    def test_denoise_failure_does_not_strand_the_work_dir(self):
        # The success path removes denoise_work/; the error path was leaving it.
        # A failure after the first sf.write would strand a full-size WAV, so the
        # cleanup has to happen wherever the job ends, not only where it succeeds.
        # Unreadable input makes denoise_file raise inside the denoise branch.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inp = td / "in.wav"
            inp.write_bytes(b"RIFFfake")
            jj = td / "job.json"
            jj.write_text(json.dumps({"description": "clean this up", "method": "denoise"}))
            out = td / "out"
            argv = ["--input", str(inp), "--job-json", str(jj), "--out-dir", str(out),
                    "--gpu-lock-timeout", "5"]
            with patch.object(clean_cli, "GPU_LOCK_PATH", td / "gpu_test.lock"), \
                 patch.object(clean_cli, "handle"):
                rc = clean_cli.main(argv)
            self.assertEqual(rc, 1)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "error")
            self.assertFalse((out / "denoise_work").exists(),
                             "denoise_work/ must not survive a failed job")
            self.assertTrue((out / "status.json").exists(),
                            "cleanup must not take out_dir itself with it")


@unittest.skipUnless(_shutil.which("ffmpeg"), "ffmpeg not on PATH")
class OpusEncodeTests(unittest.TestCase):
    def test_encodes_small_wav_to_ogg(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            wav = td / "t.wav"
            sf.write(wav, np.random.uniform(-0.3, 0.3, 16000).astype(np.float32), 16000)
            ogg = clean_cli.encode_opus(wav, td / "t.ogg")
            self.assertTrue(ogg.exists())
            self.assertGreater(ogg.stat().st_size, 0)


class WebhookTests(unittest.TestCase):
    def test_post_success(self):
        fake = MagicMock(status_code=204)
        with patch("requests.post", return_value=fake) as p:
            ok = clean_cli.post_discord_webhook("https://discord/hook", "hello")
        self.assertTrue(ok)
        self.assertEqual(p.call_args.kwargs["json"]["content"], "hello")

    def test_post_network_error_returns_false(self):
        with patch("requests.post", side_effect=OSError("net down")):
            ok = clean_cli.post_discord_webhook("https://discord/hook", "hello")
        self.assertFalse(ok)


class GpuLockTests(unittest.TestCase):
    def test_lock_blocks_second_acquirer(self):
        with clean_cli.gpu_lock(timeout_s=5):
            fh2 = open(clean_cli.GPU_LOCK_PATH, "w")
            with self.assertRaises(BlockingIOError):
                fcntl.flock(fh2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh2.close()
        # released now: acquire must succeed
        fh3 = open(clean_cli.GPU_LOCK_PATH, "w")
        fcntl.flock(fh3, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh3, fcntl.LOCK_UN)
        fh3.close()


class ProgressProtocolTests(unittest.TestCase):
    def test_emit_writes_json_line(self):
        buf = io.StringIO()
        clean_cli.emit(42, "separate", "chunk 3/45", stream=buf)
        line = json.loads(buf.getvalue().strip())
        self.assertEqual(line["type"], "progress")
        self.assertEqual(line["pct"], 42)
        self.assertEqual(line["stage"], "separate")


if __name__ == "__main__":
    unittest.main()
