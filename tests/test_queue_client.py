from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import requests

import worker.worker as w


class FakeResponse:
    def __init__(self, status_code=200, headers=None, json_payload=None):
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "application/json"}
        self._json = json_payload if json_payload is not None else {"ok": True}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


class QueueClientTests(unittest.TestCase):
    def test_request_uses_key_fallback_on_403(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = w.Config(
                server_url="https://example.com/admin/queue_worker_api.php",
                secret_key="secret",
                poll_interval=15,
                worker_id="w1",
                supported_types="sam_audio_cleanup",
                log_level="INFO",
                temp_dir=Path(td),
                heartbeat_interval=60,
                request_timeout=30,
            )
            client = w.QueueClient(cfg)

            calls = []

            def fake_request(method, url, params, timeout, **kwargs):
                calls.append(params.copy())
                if len(calls) == 1:
                    return FakeResponse(status_code=403)
                return FakeResponse(status_code=200, json_payload={"job": None})

            client.session.request = fake_request  # type: ignore[assignment]

            payload = client._request("GET", {"action": "poll"})

            self.assertEqual(payload, {"job": None})
            self.assertEqual(len(calls), 2)
            self.assertNotIn("key", calls[0])
            self.assertEqual(calls[1]["key"], "secret")


if __name__ == "__main__":
    unittest.main()
