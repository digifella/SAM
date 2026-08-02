from __future__ import annotations


class JobCancelledError(RuntimeError):
    """Raised at cancellation checkpoints to signal an operator-initiated cancel.

    Defined here (a leaf utility module) so that run_sam_interactive.py,
    worker/handlers/, and worker/worker.py can all raise/catch the same
    exception type without a circular import between the worker layer and
    the layers it calls into.
    """

    pass
