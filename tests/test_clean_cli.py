import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clean_cli


class CleanCliCoreTests(unittest.TestCase):
    def _run(self, td, extra_args=None, fake_handle=None, job_json_text=None):
        td = Path(td)
        inp = td / "in.wav"; inp.write_bytes(b"RIFFfake")
        jj = td / "job.json"
        jj.write_text(json.dumps({"description": "speech"}) if job_json_text is None else job_json_text)
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


class ProgressProtocolTests(unittest.TestCase):
    def test_emit_writes_json_line(self):
        import io
        buf = io.StringIO()
        clean_cli.emit(42, "separate", "chunk 3/45", stream=buf)
        line = json.loads(buf.getvalue().strip())
        self.assertEqual(line["type"], "progress")
        self.assertEqual(line["pct"], 42)
        self.assertEqual(line["stage"], "separate")


if __name__ == "__main__":
    unittest.main()
