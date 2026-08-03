import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import clean_cli


class CleanCliCoreTests(unittest.TestCase):
    def _run(self, td, extra_args=None, fake_handle=None):
        td = Path(td)
        inp = td / "in.wav"; inp.write_bytes(b"RIFFfake")
        jj = td / "job.json"; jj.write_text(json.dumps({"description": "speech"}))
        out = td / "out"
        argv = ["--input", str(inp), "--job-json", str(jj), "--out-dir", str(out)]
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

        with patch.object(clean_cli, "handle", side_effect=fake_handle or default_fake):
            rc = clean_cli.main(argv)
        return rc, out

    def test_success_writes_done_status_and_result_zip(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td)
            self.assertEqual(rc, 0)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "done")
            self.assertTrue((out / "result.zip").exists())

    def test_handler_error_writes_error_status_and_rc1(self):
        def boom(**kw):
            raise RuntimeError("separation exploded")
        with tempfile.TemporaryDirectory() as td:
            rc, out = self._run(td, fake_handle=boom)
            self.assertEqual(rc, 1)
            status = json.loads((out / "status.json").read_text())
            self.assertEqual(status["state"], "error")
            self.assertIn("separation exploded", status["error"])


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
