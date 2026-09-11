"""Opt-in startup diagnostics; importing this module does not require PyTorch."""

import faulthandler
import os
import sys
import time


class StartupDiagnostics:
    def __init__(self, interval=0):
        if interval < 0:
            raise ValueError("Startup debug interval must be non-negative")
        self.interval = interval
        self.active = False

    def __enter__(self):
        if self.interval:
            faulthandler.dump_traceback_later(self.interval, repeat=True, file=sys.stderr)
            self.active = True
            self.mark(f"Python stacks will be dumped every {self.interval}s until trainer initialization finishes")
        return self

    def mark(self, message):
        if self.interval:
            print(
                f"[startup rank={os.environ.get('RANK', '0')} "
                f"local_rank={os.environ.get('LOCAL_RANK', '0')} pid={os.getpid()} "
                f"time={time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}",
                file=sys.stderr, flush=True,
            )

    def probe(self, device):
        if not self.interval:
            return
        import torch

        dist = torch.distributed
        if not dist.is_initialized() or dist.get_world_size() == 1:
            self.mark(f"Communication probe skipped: no multi-rank process group; device={device}")
            return
        self.mark(f"Communication probe BEGIN: backend={dist.get_backend()}, device={device}")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        # Force the first collective before TRL's rank-zero-first tokenization block.
        value = torch.ones(1, device=device)
        dist.all_reduce(value)
        result = value.item()
        expected = dist.get_world_size()
        if result != expected:
            raise RuntimeError(f"Communication probe returned {result}; expected {expected}")
        self.mark(f"Communication probe PASSED: sum={result}")

    def stop(self):
        if self.active:
            faulthandler.cancel_dump_traceback_later()
            self.active = False

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
