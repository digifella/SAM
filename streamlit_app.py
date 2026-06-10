from __future__ import annotations

import io
import json
import os
import queue
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import streamlit as st

# Reduce CUDA allocator fragmentation for long-lived Streamlit sessions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

st.set_page_config(page_title="SAM-Audio Local Harness", page_icon="🎛️", layout="wide")


def init_state() -> None:
    ss = st.session_state
    if "job_running" not in ss:
        ss.job_running = False
    if "job_thread" not in ss:
        ss.job_thread = None
    if "job_cancel_event" not in ss:
        ss.job_cancel_event = None
    if "job_events" not in ss:
        ss.job_events = queue.Queue()
    if "progress_pct" not in ss:
        ss.progress_pct = 0
    if "status_text" not in ss:
        ss.status_text = "Idle"
    if "event_lines" not in ss:
        ss.event_lines = []
    if "run_started_at" not in ss:
        ss.run_started_at = None
    if "result_zip_bytes" not in ss:
        ss.result_zip_bytes = None
    if "result_output_data" not in ss:
        ss.result_output_data = None
    if "run_error" not in ss:
        ss.run_error = None


def append_event(line: str) -> None:
    st.session_state.event_lines.append(line)
    st.session_state.event_lines = st.session_state.event_lines[-300:]


def drain_events() -> None:
    q: queue.Queue = st.session_state.job_events
    while True:
        try:
            evt = q.get_nowait()
        except queue.Empty:
            break

        etype = evt.get("type")
        if etype == "progress":
            pct = max(0, min(100, int(evt.get("pct", 0))))
            stage = str(evt.get("stage") or "processing")
            msg = str(evt.get("message") or "")
            ts = str(evt.get("ts") or time.strftime("%H:%M:%S"))
            st.session_state.progress_pct = pct
            st.session_state.status_text = f"[{stage}] {msg}"
            append_event(f"{ts} | {pct:>3}% | {stage:<12} | {msg}")
        elif etype == "done":
            st.session_state.progress_pct = 100
            st.session_state.status_text = "Processing complete"
            st.session_state.result_output_data = evt.get("output_data")
            st.session_state.result_zip_bytes = evt.get("zip_bytes")
            st.session_state.run_error = None
            append_event(f"{time.strftime('%H:%M:%S')} | 100% | complete      | Processing complete")
        elif etype == "error":
            err = str(evt.get("error") or "Unknown error")
            st.session_state.run_error = err
            st.session_state.status_text = f"Error: {err}"
            append_event(f"{time.strftime('%H:%M:%S')} | ERR | exception    | {err}")
        elif etype == "finished":
            st.session_state.job_running = False
            st.session_state.job_thread = None
            st.session_state.job_cancel_event = None


def worker_run(upload_name: str, upload_bytes: bytes, payload: dict, events: queue.Queue, cancel_event: threading.Event) -> None:
    def push(event: dict) -> None:
        events.put(event)

    def progress_cb(progress_pct: float, message: str, stage: Optional[str] = None) -> None:
        push(
            {
                "type": "progress",
                "pct": int(progress_pct),
                "message": str(message or ""),
                "stage": str(stage or "processing"),
                "ts": time.strftime("%H:%M:%S"),
            }
        )

    try:
        from worker.handlers.sam_audio_cleanup import handle

        with tempfile.TemporaryDirectory(prefix="sam_streamlit_") as td:
            work_dir = Path(td)
            input_path = work_dir / upload_name
            input_path.write_bytes(upload_bytes)

            result = handle(
                input_path=input_path,
                input_data=payload,
                job={"id": 0, "type": "sam_audio_cleanup", "input_filename": upload_name},
                progress_cb=progress_cb,
                is_cancelled_cb=lambda: cancel_event.is_set(),
            )

            output_data = result.get("output_data", {}) if isinstance(result, dict) else {}
            output_file = Path(result.get("output_file")) if isinstance(result, dict) and result.get("output_file") else None
            if not output_file or not output_file.exists():
                raise RuntimeError("Handler did not return a valid output ZIP")

            zip_bytes = output_file.read_bytes()
            push({"type": "done", "output_data": output_data, "zip_bytes": zip_bytes})
    except Exception as e:
        if cancel_event.is_set():
            push({"type": "error", "error": f"Cancelled: {e}"})
        else:
            push({"type": "error", "error": str(e)})
    finally:
        push({"type": "finished"})


def latest_log_tail(max_lines: int = 200) -> tuple[str, str]:
    log_dir = Path.home() / ".sam_audio_logs"
    if not log_dir.exists():
        return "", "No ~/.sam_audio_logs directory yet."

    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return "", "No log files found yet."

    newest = logs[0]
    text = newest.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    return newest.name, tail or "(log file is empty)"


def clear_runtime_logs() -> int:
    log_dir = Path.home() / ".sam_audio_logs"
    if not log_dir.exists():
        return 0
    removed = 0
    for p in log_dir.glob("sam_audio_*.log"):
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    return removed


def main() -> None:
    init_state()
    drain_events()

    st.title("SAM-Audio Local Harness")
    st.caption("Runs the same `sam_audio_cleanup` handler used by the queue worker, with stop control and live logs.")

    with st.sidebar:
        st.header("Run Options")
        description = st.text_input("What to extract", value="speech")
        convert_to_mono = st.checkbox("Convert to mono 16k first", value=True)
        chunk_duration = st.number_input("Chunk duration (s)", min_value=5, max_value=600, value=60, step=5)
        overlap = st.number_input("Chunk overlap (s)", min_value=0.0, max_value=30.0, value=2.0, step=0.5)
        rerank = st.slider("Rerank candidates", min_value=1, max_value=8, value=1)
        predict_spans = st.checkbox("Predict spans", value=False)

        st.subheader("Optional transforms")
        trial_seconds = st.number_input("Trial only first N seconds (0 = full)", min_value=0, max_value=86400, value=0, step=5)
        normalize_percent = st.number_input("Normalize peak (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)

        sample_rate_choice = st.selectbox(
            "Output sample rate",
            options=["No change", "16000", "32000", "44100", "48000"],
            index=0,
        )
        channels_choice = st.selectbox("Output channels", options=["No change", "Mono (1)", "Stereo (2)"], index=0)

        st.subheader("Runtime")
        model_dir_default = os.environ.get("SAM_MODEL_DIR", str(Path.home() / "models" / "sam-audio-large-tv"))
        model_dir = st.text_input("Model directory", value=model_dir_default)
        device = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
        memory_fraction = st.slider("CUDA memory fraction", min_value=0.10, max_value=0.98, value=0.85, step=0.01)
        allow_cpu_fallback = st.checkbox("Auto-fallback to CPU on persistent CUDA OOM", value=True)
        clear_logs_on_start = st.checkbox("Clear old ~/.sam_audio_logs on each new run", value=True)

    uploaded = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
        accept_multiple_files=False,
    )

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        start_disabled = st.session_state.job_running or uploaded is None
        start_clicked = st.button("Start Job", type="primary", disabled=start_disabled)
    with col_b:
        stop_clicked = st.button("Stop Job", disabled=not st.session_state.job_running)

    if start_clicked and uploaded is not None:
        payload: dict = {
            "description": description,
            "convert_to_mono": bool(convert_to_mono),
            "chunk_duration": int(chunk_duration),
            "overlap": float(overlap),
            "rerank": int(rerank),
            "predict_spans": bool(predict_spans),
            "trial_seconds": int(trial_seconds),
            "normalize_percent": float(normalize_percent),
            "model_dir": model_dir,
            "device": device,
            "memory_fraction": float(memory_fraction),
            "allow_cpu_fallback": bool(allow_cpu_fallback),
        }

        if sample_rate_choice != "No change":
            payload["output_sample_rate"] = int(sample_rate_choice)

        if channels_choice == "Mono (1)":
            payload["output_channels"] = 1
        elif channels_choice == "Stereo (2)":
            payload["output_channels"] = 2

        st.session_state.result_zip_bytes = None
        st.session_state.result_output_data = None
        st.session_state.run_error = None
        st.session_state.progress_pct = 0
        st.session_state.status_text = "Starting job"
        st.session_state.event_lines = []
        st.session_state.run_started_at = time.time()
        if clear_logs_on_start:
            removed = clear_runtime_logs()
            append_event(f"{time.strftime('%H:%M:%S')} | ... | cleanup       | Removed {removed} old runtime log(s)")
        append_event(f"{time.strftime('%H:%M:%S')} |   0% | start         | Job submitted from UI")

        cancel_event = threading.Event()
        th = threading.Thread(
            target=worker_run,
            args=(uploaded.name, uploaded.getvalue(), payload, st.session_state.job_events, cancel_event),
            daemon=True,
            name="sam-audio-ui-worker",
        )
        st.session_state.job_cancel_event = cancel_event
        st.session_state.job_thread = th
        st.session_state.job_running = True
        th.start()

    if stop_clicked and st.session_state.job_running and st.session_state.job_cancel_event is not None:
        st.session_state.job_cancel_event.set()
        append_event(f"{time.strftime('%H:%M:%S')} | ... | cancel        | Stop requested by operator")
        st.session_state.status_text = "Cancellation requested"

    st.subheader("Run Progress")
    st.progress(int(st.session_state.progress_pct))
    if st.session_state.run_started_at:
        elapsed = time.time() - float(st.session_state.run_started_at)
        st.caption(f"Status: {st.session_state.status_text} | Elapsed: {elapsed:.1f}s")
    else:
        st.caption(f"Status: {st.session_state.status_text}")

    st.code("\n".join(st.session_state.event_lines) or "No events yet.", language="text")

    if st.session_state.run_error:
        st.error(st.session_state.run_error)
        log_name, log_tail = latest_log_tail(max_lines=180)
        st.markdown("**Latest runtime log excerpt**")
        if log_name:
            st.caption(log_name)
        st.code(log_tail, language="text")

    if st.session_state.result_output_data:
        st.subheader("Metadata")
        st.json(st.session_state.result_output_data)

    if st.session_state.result_zip_bytes:
        zip_bytes = st.session_state.result_zip_bytes
        st.download_button(
            label="Download ZIP Result",
            data=zip_bytes,
            file_name="audio_cleanup_result.zip",
            mime="application/zip",
        )

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = set(zf.namelist())
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Target**")
                if "target.wav" in names:
                    st.audio(zf.read("target.wav"), format="audio/wav")
                else:
                    st.caption("target.wav not found in ZIP")
            with col2:
                st.markdown("**Residual**")
                if "residual.wav" in names:
                    st.audio(zf.read("residual.wav"), format="audio/wav")
                else:
                    st.caption("residual.wav not found in ZIP")

            if "metadata.json" in names:
                with st.expander("metadata.json"):
                    try:
                        st.code(json.dumps(json.loads(zf.read("metadata.json").decode("utf-8")), indent=2), language="json")
                    except Exception:
                        st.code(zf.read("metadata.json").decode("utf-8", errors="ignore"))

    with st.expander("Latest SAM Runtime Log (~/.sam_audio_logs)", expanded=False):
        name, tail = latest_log_tail(max_lines=250)
        if name:
            st.caption(f"Latest log: {name}")
        st.code(tail, language="text")

    if st.session_state.job_running:
        # Keep UI moving while background worker runs.
        time.sleep(1)
        st.experimental_rerun()


if __name__ == "__main__":
    main()
