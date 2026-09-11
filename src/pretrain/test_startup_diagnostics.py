"""Run without ML dependencies: python -m unittest discover -s src/pretrain -p test_startup_diagnostics.py."""

from contextlib import redirect_stderr
import io
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from startup_diagnostics import StartupDiagnostics


class StartupDiagnosticsTest(unittest.TestCase):
    def test_disabled_is_noop(self):
        with patch("startup_diagnostics.faulthandler") as handler, redirect_stderr(io.StringIO()) as log:
            with StartupDiagnostics() as debug:
                debug.mark("hidden")
                debug.probe(None)
                debug.stop()
            self.assertEqual(log.getvalue(), "")
            self.assertEqual(handler.mock_calls, [])

    def test_enabled_timer_and_rank_markers(self):
        with patch("startup_diagnostics.faulthandler") as handler, \
                patch.dict("os.environ", {"RANK": "1", "LOCAL_RANK": "1"}), \
                redirect_stderr(io.StringIO()) as log:
            with StartupDiagnostics(60) as debug:
                debug.mark("Trainer initialization BEGIN")
                debug.stop()
            handler.dump_traceback_later.assert_called_once_with(60, repeat=True, file=log)
            handler.cancel_dump_traceback_later.assert_called_once_with()
            self.assertIn("rank=1 local_rank=1", log.getvalue())
            self.assertIn("Trainer initialization BEGIN", log.getvalue())

    def test_exception_cancels_timer_without_swallowing_error(self):
        with patch("startup_diagnostics.faulthandler") as handler, redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                with StartupDiagnostics(60):
                    raise RuntimeError("startup failed")
            handler.cancel_dump_traceback_later.assert_called_once_with()

    def test_rejects_negative_interval(self):
        with self.assertRaises(ValueError):
            StartupDiagnostics(-1)

    def test_probe_passes_on_expected_sum(self):
        for device_type in ["cpu", "cuda"]:
            with self.subTest(device=device_type):
                torch = MagicMock()
                torch.distributed.is_initialized.return_value = True
                torch.distributed.get_world_size.return_value = 2
                torch.distributed.get_backend.return_value = "nccl" if device_type == "cuda" else "gloo"
                torch.ones.return_value.item.return_value = 2.0
                device = SimpleNamespace(type=device_type)
                with patch.dict("sys.modules", {"torch": torch}), redirect_stderr(io.StringIO()) as log:
                    StartupDiagnostics(60).probe(device)
                torch.ones.assert_called_once_with(1, device=device)
                torch.distributed.all_reduce.assert_called_once_with(torch.ones.return_value)
                if device_type == "cuda":
                    torch.cuda.set_device.assert_called_once_with(device)
                else:
                    torch.cuda.set_device.assert_not_called()
                self.assertIn("Communication probe PASSED", log.getvalue())

    def test_probe_skips_without_multiple_ranks(self):
        for initialized, size in [(False, 2), (True, 1)]:
            torch = MagicMock()
            torch.distributed.is_initialized.return_value = initialized
            torch.distributed.get_world_size.return_value = size
            with patch.dict("sys.modules", {"torch": torch}), redirect_stderr(io.StringIO()):
                StartupDiagnostics(60).probe(SimpleNamespace(type="cpu"))
            torch.distributed.all_reduce.assert_not_called()

    def test_probe_rejects_wrong_sum(self):
        torch = MagicMock()
        torch.distributed.is_initialized.return_value = True
        torch.distributed.get_world_size.return_value = 2
        torch.ones.return_value.item.return_value = 1.0
        with patch.dict("sys.modules", {"torch": torch}), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "expected 2"):
                StartupDiagnostics(60).probe(SimpleNamespace(type="cpu"))


if __name__ == "__main__":
    unittest.main()
