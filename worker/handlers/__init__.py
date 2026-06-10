from __future__ import annotations

import importlib
from typing import Callable, Dict

_HANDLER_SPECS: Dict[str, str] = {
    "sam_audio_cleanup": "worker.handlers.sam_audio_cleanup:handle",
}


def get_handler(job_type: str) -> Callable | None:
    spec = _HANDLER_SPECS.get(job_type)
    if not spec:
        return None
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name, None)


def supported_types() -> list[str]:
    return sorted(_HANDLER_SPECS.keys())
