import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import worker.handlers.sam_audio_cleanup as h


def _fake_process(audio_path, description, output_dir, **kw):
    sr = 16000
    noise = np.random.uniform(-0.5, 0.5, sr).astype(np.float32)
    sf.write(Path(output_dir) / "in_target.wav", noise, sr)
    sf.write(Path(output_dir) / "in_residual.wav", noise, sr)
    return True


class HandlerWorkDirTests(unittest.TestCase):
    def test_handle_uses_caller_work_dir_and_never_mkdtemps(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            input_wav = td / "in.wav"
            sf.write(input_wav, np.zeros(16000, dtype=np.float32), 16000)

            with patch.object(h._ModelCache, "get", return_value=(None, None, "cpu")), \
                 patch.object(h, "process_audio_file", side_effect=_fake_process), \
                 patch.object(h, "auto_input_gain",
                              side_effect=lambda p, f, out_dir=None: (p, 0.0)), \
                 patch.object(h, "get_audio_duration", return_value=1.0), \
                 patch.object(h, "count_chunks", return_value=1), \
                 patch.object(h.tempfile, "mkdtemp",
                              side_effect=AssertionError("mkdtemp must not be called")):
                result = h.handle(
                    input_path=input_wav,
                    input_data={"description": "speech"},
                    job={"id": 1},
                    work_dir=td,
                )

            out_zip = Path(result["output_file"])
            self.assertTrue(out_zip.exists())
            self.assertTrue(str(out_zip).startswith(str(td)),
                            f"{out_zip} not under caller work_dir {td}")


if __name__ == "__main__":
    unittest.main()
