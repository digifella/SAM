"""F6: process_chunk's except block must release target_tensor/residual_tensor
on the error path, the same way it already releases batch/result, so a
retained exception (worker/handlers/sam_audio_cleanup.py stores it in
last_error for the whole retry ladder) does not keep GPU tensors reachable
through its __traceback__ frame chain.

This test does not require a GPU: it triggers the failure with a real CPU
torch.Tensor for target and a fake object that raises on .squeeze() for
residual, so both target_tensor and residual_tensor get bound in the frame
(mirroring the real code path, where both are assigned together before
either .squeeze() call) before the exception fires. It then walks the raised
exception's traceback frame chain and asserts neither name is still bound in
process_chunk's frame -- i.e. `del` actually removed the binding, so the
traceback no longer keeps the tensor reachable.
"""

from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock

import numpy as np
import torch

import run_sam_interactive as r


class _RaisingOnSqueeze:
    """Stands in for a GPU residual tensor whose .squeeze() blows up, so the
    except block fires after both target_tensor and residual_tensor are
    already bound (matching the real assignment order in process_chunk)."""

    def squeeze(self):
        raise RuntimeError("simulated failure during residual post-processing")


def _find_frame(tb, func_name):
    while tb is not None:
        if tb.tb_frame.f_code.co_name == func_name:
            return tb.tb_frame
        tb = tb.tb_next
    return None


class ProcessChunkErrorPathTensorReleaseTests(unittest.TestCase):
    def test_target_and_residual_tensor_unbound_in_frame_after_error(self):
        processor = MagicMock()
        processor.return_value.to.return_value = "fake_batch"

        model = MagicMock()
        result = MagicMock()
        result.target = [torch.zeros(4)]
        result.residual = [_RaisingOnSqueeze()]
        model.separate.return_value = result

        chunk_data = np.zeros(1600, dtype=np.float32)

        # NOTE: deliberately not using self.assertRaises here -- its __exit__
        # calls traceback.clear_frames() and detaches the traceback entirely
        # (exc_value.with_traceback(None)) before storing the exception,
        # which would clear every frame's locals regardless of whether our
        # fix works and defeat this test. A plain try/except preserves the
        # real, live traceback exactly as the production retry ladder sees it.
        exc = None
        try:
            # process_chunk's (pre-existing, untouched-by-this-fix) inference
            # block uses the deprecated torch.cuda.amp.autocast spelling;
            # this test is simply the first in the suite to exercise that
            # line directly. Filter it locally rather than edit unrelated
            # source, so the suite's warning count doesn't grow.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*"
                )
                r.process_chunk(
                    chunk_data=chunk_data,
                    sample_rate=16000,
                    description="x",
                    model=model,
                    processor=processor,
                    device="cpu",
                    rerank=1,
                    predict_spans=False,
                )
        except RuntimeError as e:
            exc = e

        self.assertIsNotNone(exc, "process_chunk did not raise")
        frame = _find_frame(exc.__traceback__, "process_chunk")
        self.assertIsNotNone(frame, "process_chunk frame not found in traceback")
        self.assertNotIn(
            "target_tensor",
            frame.f_locals,
            "target_tensor still bound in process_chunk's frame after the "
            "error path -- the traceback is keeping a GPU tensor reachable",
        )
        self.assertNotIn(
            "residual_tensor",
            frame.f_locals,
            "residual_tensor still bound in process_chunk's frame after the "
            "error path -- the traceback is keeping a GPU tensor reachable",
        )


if __name__ == "__main__":
    unittest.main()
