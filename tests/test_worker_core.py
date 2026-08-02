from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import worker.worker as w
from sam_audio_utils.errors import JobCancelledError as SharedJobCancelledError


class DummyClient:
    def __init__(self):
        self.completed = []
        self.failed = []
        self.downloaded = []

    def download_input(self, job_id: int, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"audio")
        self.downloaded.append((job_id, out_path))
        return out_path

    def complete(self, job_id: int, output_data: dict, output_file: Path | None):
        self.completed.append((job_id, output_data, output_file))

    def fail(self, job_id: int, error_message: str):
        self.failed.append((job_id, error_message))

    def heartbeat(self, job_id: int):
        return None


class FailingCompleteClient(DummyClient):
    """DummyClient whose complete() always raises, to simulate an upload failure."""

    def complete(self, job_id: int, output_data: dict, output_file: Path | None):
        raise ConnectionError("simulated upload blip")


class WorkerCoreTests(unittest.TestCase):
    def test_parse_input_data_variants(self):
        self.assertEqual(w.parse_input_data(None), {})
        self.assertEqual(w.parse_input_data(""), {})
        self.assertEqual(w.parse_input_data("not-json"), {})
        self.assertEqual(w.parse_input_data('{"a":1}'), {"a": 1})
        self.assertEqual(w.parse_input_data({"x": "y"}), {"x": "y"})

    def test_process_job_success(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_file = td_path / "result.zip"
            output_file.write_bytes(b"zip")

            def fake_handler(input_path, input_data, job, progress_cb=None):
                self.assertTrue(input_path.exists())
                self.assertEqual(input_data, {"description": "speech"})
                if progress_cb:
                    progress_cb(50, "half", "processing")
                return {"output_data": {"ok": True}, "output_file": output_file}

            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = DummyClient()
            job = {
                "id": 7,
                "type": "sam_audio_cleanup",
                "input_filename": "input.wav",
                "input_data": json.dumps({"description": "speech"}),
            }

            with patch.object(w, "get_handler", return_value=fake_handler):
                w.process_job(client, cfg, job)

            self.assertEqual(len(client.failed), 0)
            self.assertEqual(len(client.completed), 1)
            jid, output_data, returned_file = client.completed[0]
            self.assertEqual(jid, 7)
            self.assertTrue(output_data["ok"])
            self.assertEqual(returned_file, output_file)

    def test_process_job_passes_work_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            received = {}

            def fake_handler(input_path, input_data, job, progress_cb=None, work_dir=None):
                received["work_dir"] = work_dir
                return {"output_data": {"ok": True}, "output_file": None}

            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = DummyClient()
            job = {
                "id": 7,
                "type": "sam_audio_cleanup",
                "input_filename": "",
                "input_data": "{}",
            }

            with patch.object(w, "get_handler", return_value=fake_handler):
                w.process_job(client, cfg, job)

            self.assertEqual(len(client.failed), 0)
            self.assertIsNotNone(received["work_dir"])
            self.assertTrue(str(received["work_dir"]).startswith(str(cfg.temp_dir)))
            # F3's finally: only skips rmtree when the upload failed; on a
            # normal success it must still delete work_dir, or every
            # successful job leaks a directory forever.
            self.assertFalse(received["work_dir"].exists())

    def test_process_job_upload_failure_preserves_work_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_file = td_path / "result.zip"
            output_file.write_bytes(b"zip")
            received = {}

            def fake_handler(input_path, input_data, job, progress_cb=None, work_dir=None):
                received["work_dir"] = work_dir
                return {"output_data": {"ok": True}, "output_file": output_file}

            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = FailingCompleteClient()
            job = {
                "id": 11,
                "type": "sam_audio_cleanup",
                "input_filename": "",
                "input_data": "{}",
            }

            with patch.object(w, "get_handler", return_value=fake_handler):
                w.process_job(client, cfg, job)

            # Upload failure must not be reported as a processing failure.
            self.assertEqual(client.failed, [])
            self.assertEqual(client.completed, [])
            # And the computed output must survive so it is recoverable.
            self.assertIsNotNone(received["work_dir"])
            self.assertTrue(received["work_dir"].exists())

    def test_process_job_mkdtemp_failure_fails_job(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = DummyClient()
            job = {"id": 13, "type": "sam_audio_cleanup", "input_data": "{}"}

            def fake_handler(input_path, input_data, job, progress_cb=None):
                raise AssertionError("handler must not run when work dir setup fails")

            with patch.object(w, "get_handler", return_value=fake_handler), \
                 patch.object(w.tempfile, "mkdtemp", side_effect=PermissionError("no perm")):
                w.process_job(client, cfg, job)  # must not raise

            self.assertEqual(len(client.failed), 1)
            self.assertEqual(client.failed[0][0], 13)
            self.assertEqual(client.completed, [])

    def test_process_job_heartbeat_start_failure_fails_job_and_cleans_up(self):
        # F4 covers two failure points under the same try: mkdtemp (above) and
        # hb.start(). Here mkdtemp succeeds (so a work_dir is actually
        # created) but hb.start() raises, to prove the leaked-directory half
        # of F4 is also covered, not just the mkdtemp half.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = DummyClient()
            job = {"id": 17, "type": "sam_audio_cleanup", "input_data": "{}"}

            def fake_handler(input_path, input_data, job, progress_cb=None):
                raise AssertionError("handler must not run when heartbeat setup fails")

            with patch.object(w, "get_handler", return_value=fake_handler), \
                 patch.object(w.HeartbeatThread, "start", side_effect=RuntimeError("thread start failed")):
                w.process_job(client, cfg, job)  # must not raise

            self.assertEqual(len(client.failed), 1)
            self.assertEqual(client.failed[0][0], 17)
            # The fail message must be hb.start()'s own error, not the
            # handler's assertion -- proving the job was failed during setup
            # and the handler genuinely never ran.
            self.assertIn("thread start failed", client.failed[0][1])
            self.assertEqual(client.completed, [])
            # The work_dir mkdtemp created before hb.start() raised must not leak.
            self.assertEqual(list(cfg.temp_dir.iterdir()), [])

    def test_process_job_unsupported_type(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            client = DummyClient()

            with patch.object(w, "get_handler", return_value=None):
                w.process_job(client, cfg, {"id": 9, "type": "unknown", "input_data": "{}"})

            self.assertEqual(client.completed, [])
            self.assertEqual(len(client.failed), 1)
            self.assertIn("Unsupported job type", client.failed[0][1])

    def test_job_cancelled_error_is_the_shared_class(self):
        # worker.worker.JobCancelledError must be a re-export of the shared
        # sam_audio_utils.errors class, not a second, unrelated class of the
        # same name (which would silently defeat the except-clause match).
        self.assertIs(w.JobCancelledError, SharedJobCancelledError)

    def test_process_job_cancellation_uses_cancelled_by_operator_path(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)

            def fake_handler(input_path, input_data, job, progress_cb=None):
                raise w.JobCancelledError("Cancelled during post-processing")

            cfg = w.Config(
                server_url="https://example.com",
                secret_key="x",
                poll_interval=1,
                worker_id="test",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=td_path / "tmp",
                heartbeat_interval=60,
                request_timeout=10,
            )
            cfg.temp_dir.mkdir(parents=True, exist_ok=True)

            client = DummyClient()
            job = {"id": 21, "type": "sam_audio_cleanup", "input_data": "{}"}

            with patch.object(w, "get_handler", return_value=fake_handler):
                w.process_job(client, cfg, job)

            # Reported via the "Cancelled by operator" path, not the generic
            # failure path.
            self.assertEqual(client.completed, [])
            self.assertEqual(len(client.failed), 1)
            self.assertIn("Cancelled by operator", client.failed[0][1])
            self.assertIn("Cancelled during post-processing", client.failed[0][1])


if __name__ == "__main__":
    unittest.main()
