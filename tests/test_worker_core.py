from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import worker.worker as w


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


if __name__ == "__main__":
    unittest.main()
