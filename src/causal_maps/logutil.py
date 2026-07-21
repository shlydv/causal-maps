"""Timestamped heartbeat logging + progress.json for Kaggle runs.

Why this exists: Kaggle's API does NOT stream a running kernel's stdout — we get
the full log only when the kernel finishes. So we (a) print frequent timestamped
progress lines with flush=True (the committed log then proves the run was never
stuck and shows steady progress), and (b) write a progress.json snapshot to the
output dir on each heartbeat (retrievable with the committed output).
"""
import json
import os
import time


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class Heartbeat:
    """Progress tracker. Call .step() once per unit of work; it prints at most
    once per `every_sec` (plus always on the first and last step)."""

    def __init__(self, total, stage, every_sec=30.0, out_dir=None):
        self.total = max(1, int(total))
        self.stage = stage
        self.every = float(every_sec)
        self.out_dir = out_dir
        self.t0 = time.time()
        self.last = 0.0
        self.i = 0
        log(f"START stage={stage} total={self.total}")

    def step(self, i=None, extra=""):
        self.i = (self.i + 1) if i is None else int(i)
        now = time.time()
        is_last = self.i >= self.total
        if (now - self.last) >= self.every or is_last or self.i == 1:
            self.last = now
            elapsed = now - self.t0
            frac = self.i / self.total
            # clamp so an undercounted `total` can't show >100% or negative eta
            pct = min(100.0, 100.0 * frac)
            eta = (elapsed / frac - elapsed) if 0 < frac <= 1 else 0.0
            log(f"stage={self.stage} {self.i}/{self.total} ({pct:.1f}%) "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s {extra}")
            self._write_progress(elapsed, eta, pct, extra)

    def _write_progress(self, elapsed, eta, pct, extra):
        if not self.out_dir:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            path = os.path.join(self.out_dir, "progress.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({
                    "stage": self.stage, "i": self.i, "total": self.total,
                    "pct": pct, "elapsed_s": elapsed, "eta_s": eta,
                    "extra": extra, "ts": time.time(),
                }, f)
            os.replace(tmp, path)
        except Exception as e:  # never let logging kill a run
            log(f"WARN progress.json write failed: {e}")

    def done(self, extra=""):
        elapsed = time.time() - self.t0
        log(f"DONE stage={self.stage} {self.i}/{self.total} "
            f"elapsed={elapsed:.0f}s {extra}")
