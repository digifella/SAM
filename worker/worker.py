#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __package__ in (None, ""):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from handlers import get_handler
else:
    from .handlers import get_handler


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


@dataclass
class Config:
    server_url: str
    secret_key: str
    poll_interval: int
    worker_id: str
    supported_types: str
    log_level: str
    temp_dir: Path
    heartbeat_interval: int
    request_timeout: int


def read_config(config_path: Optional[Path] = None) -> Config:
    env_path = config_path or (ROOT / "worker" / "config.env")
    file_vars = load_env_file(env_path)

    def get(name: str, default: str) -> str:
        return os.environ.get(name, file_vars.get(name, default))

    cfg = Config(
        server_url=get("QUEUE_SERVER_URL", "").strip(),
        secret_key=get("QUEUE_SECRET_KEY", "").strip(),
        poll_interval=int(get("POLL_INTERVAL", "15")),
        worker_id=get("WORKER_ID", "sam-worker-1").strip(),
        supported_types=get("SUPPORTED_TYPES", "sam_audio_cleanup").strip(),
        log_level=get("LOG_LEVEL", "INFO").strip().upper(),
        temp_dir=Path(get("TEMP_DIR", str(ROOT / "worker" / "tmp"))),
        heartbeat_interval=int(get("HEARTBEAT_INTERVAL", "60")),
        request_timeout=int(get("REQUEST_TIMEOUT", "90")),
    )

    if not cfg.server_url or not cfg.secret_key:
        raise RuntimeError("Missing QUEUE_SERVER_URL or QUEUE_SECRET_KEY in worker/config.env or environment")
    return cfg


class QueueClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"X-Queue-Key": cfg.secret_key})

    def _request(self, method: str, params: Dict[str, str], **kwargs):
        query = dict(params)
        response = self.session.request(
            method=method,
            url=self.cfg.server_url,
            params=query,
            timeout=self.cfg.request_timeout,
            **kwargs,
        )
        if response.status_code == 403 and "key" not in query:
            fallback_query = dict(query)
            fallback_query["key"] = self.cfg.secret_key
            response = self.session.request(
                method=method,
                url=self.cfg.server_url,
                params=fallback_query,
                timeout=self.cfg.request_timeout,
                **kwargs,
            )
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "").lower()
        if "application/json" in ctype:
            return response.json()
        return response

    def poll(self) -> Optional[dict]:
        payload = self._request(
            "GET",
            {
                "action": "poll",
                "types": self.cfg.supported_types,
                "worker_id": self.cfg.worker_id,
            },
        )
        return payload.get("job") if isinstance(payload, dict) else None

    def download_input(self, job_id: int, out_path: Path) -> Optional[Path]:
        response = self._request("GET", {"action": "download_input", "id": str(job_id)})
        if not isinstance(response, requests.Response):
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
        return out_path

    def heartbeat(self, job_id: int) -> None:
        self._request("POST", {"action": "heartbeat", "id": str(job_id)})

    def fail(self, job_id: int, error_message: str) -> None:
        self._request(
            "POST",
            {"action": "fail", "id": str(job_id)},
            data={"error": error_message[:5000]},
        )

    def complete(self, job_id: int, output_data: dict, output_file: Optional[Path]) -> None:
        data = {"output_data": json.dumps(output_data or {})}
        files = None
        if output_file and output_file.exists():
            files = {"file": (output_file.name, open(output_file, "rb"), "application/octet-stream")}
        try:
            self._request(
                "POST",
                {"action": "complete", "id": str(job_id)},
                data=data,
                files=files,
            )
        finally:
            if files:
                files["file"][1].close()


class HeartbeatThread(threading.Thread):
    def __init__(self, client: QueueClient, job_id: int, interval: int):
        super().__init__(daemon=True)
        self.client = client
        self.job_id = job_id
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.wait(self.interval):
            try:
                self.client.heartbeat(self.job_id)
            except Exception:
                logging.exception("Heartbeat failed for job %s", self.job_id)


class JobCancelledError(RuntimeError):
    pass


def parse_input_data(raw_value) -> dict:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def process_job(client: QueueClient, cfg: Config, job: dict) -> None:
    job_id = int(job["id"])
    job_type = str(job.get("type", ""))

    handler = get_handler(job_type)
    if handler is None:
        client.fail(job_id, f"Unsupported job type: {job_type}")
        return

    input_data = parse_input_data(job.get("input_data"))

    logging.info("Claimed job id=%s type=%s", job_id, job_type)
    work_dir = Path(tempfile.mkdtemp(prefix=f"queue_job_{job_id}_", dir=str(cfg.temp_dir)))
    input_path: Optional[Path] = None

    hb = HeartbeatThread(client, job_id, cfg.heartbeat_interval)
    hb.start()
    try:
        input_filename = str(job.get("input_filename", "") or "")
        if input_filename:
            input_path = work_dir / input_filename
            downloaded = client.download_input(job_id, input_path)
            if downloaded is None:
                raise RuntimeError("Failed to download input file")
            input_path = downloaded
            logging.info("Downloaded input to %s", input_path)

        def _progress_cb(progress_pct: float, message: str, stage: Optional[str] = None) -> None:
            pct = max(0, min(100, int(progress_pct)))
            stage_text = stage or "processing"
            logging.info("job=%s progress=%s%% stage=%s msg=%s", job_id, pct, stage_text, message)

        handler_kwargs = {"input_path": input_path, "input_data": input_data, "job": job}
        handler_params = set(inspect.signature(handler).parameters.keys())
        if "progress_cb" in handler_params:
            handler_kwargs["progress_cb"] = _progress_cb

        result = handler(**handler_kwargs)
        output_data = result.get("output_data", {}) if isinstance(result, dict) else {}
        output_file = result.get("output_file") if isinstance(result, dict) else None
        output_file = Path(output_file) if output_file else None
        if not isinstance(output_data, dict):
            output_data = {"result": output_data}

        client.complete(job_id, output_data=output_data, output_file=output_file)
        logging.info("Completed job id=%s", job_id)
    except JobCancelledError as e:
        logging.warning("Job id=%s cancelled: %s", job_id, e)
        try:
            client.fail(job_id, f"Cancelled by operator: {str(e)}")
        except Exception:
            logging.exception("Failed to submit cancel status for job id=%s", job_id)
    except Exception as e:
        logging.exception("Job id=%s failed", job_id)
        try:
            client.fail(job_id, str(e))
        except Exception:
            logging.exception("Failed to submit fail status for job id=%s", job_id)
    finally:
        hb.stop()
        hb.join(timeout=2)
        shutil.rmtree(work_dir, ignore_errors=True)


def run_worker(cfg: Config) -> int:
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        "Queue worker started: worker_id=%s types=%s poll=%ss",
        cfg.worker_id,
        cfg.supported_types,
        cfg.poll_interval,
    )

    client = QueueClient(cfg)
    stop_event = threading.Event()

    def _stop(*_args):
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    consecutive_conn_errors = 0
    while not stop_event.is_set():
        try:
            job = client.poll()
            if consecutive_conn_errors > 0:
                logging.info("Queue connectivity restored after %s failed attempt(s)", consecutive_conn_errors)
            consecutive_conn_errors = 0

            if not job:
                time.sleep(cfg.poll_interval)
                continue

            process_job(client, cfg, job)
        except requests.ConnectionError as e:
            consecutive_conn_errors += 1
            backoff_seconds = min(max(cfg.poll_interval, 5) * (2 ** min(consecutive_conn_errors - 1, 5)), 300)
            logging.warning(
                "Queue server unreachable (attempt=%s, retry_in=%ss): %s",
                consecutive_conn_errors,
                int(backoff_seconds),
                str(e)[:240],
            )
            time.sleep(backoff_seconds)
        except requests.Timeout as e:
            consecutive_conn_errors += 1
            backoff_seconds = min(max(cfg.poll_interval, 5) * (2 ** min(consecutive_conn_errors - 1, 5)), 300)
            logging.warning(
                "Queue request timeout (attempt=%s, retry_in=%ss): %s",
                consecutive_conn_errors,
                int(backoff_seconds),
                str(e)[:240],
            )
            time.sleep(backoff_seconds)
        except Exception:
            logging.exception("Worker loop error")
            time.sleep(cfg.poll_interval)

    logging.info("Worker stopping")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM-Audio queue worker")
    parser.add_argument("--config", type=str, default="", help="Path to env config file")
    args = parser.parse_args()

    cfg = read_config(Path(args.config) if args.config else None)
    return run_worker(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
