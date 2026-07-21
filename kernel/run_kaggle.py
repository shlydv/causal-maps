#!/usr/bin/env python3
"""Local orchestrator: ship code to Kaggle, run a stage on GPU, monitor, pull.

Runs on the laptop (stdlib only — no torch). Drives the kaggle CLI.

  push-code   zip src/ and create/version the `causal-maps-code` dataset
  smoke       push a tiny GPU kernel to validate the whole pipeline
  run STAGE   push a kernel that runs causal_maps.experiments.main(STAGE)
  status SLUG / output SLUG   inspect a kernel

Monitoring: after push we poll `kernels status` every --poll seconds, printing a
timestamped line each time (liveness). Kaggle's API does not stream a running
kernel's stdout, so we (a) show RUNNING/COMPLETE/ERROR live, (b) warn if the run
exceeds --max-wait (possible stall), and (c) pull + tail the full log at the end
so the committed heartbeat lines prove steady progress.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile

def _read_username():
    cfg = os.path.expanduser(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"))
    with open(os.path.join(cfg, "kaggle.json")) as f:
        return json.load(f)["username"]


USER = _read_username()  # from the active kaggle.json (robust to account swaps)
CODE_SLUG = f"{USER}/causal-maps-code"
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJ, "src")
WORK = os.path.join(PROJ, "kernel", "_work")
RUNS = os.path.join(PROJ, "runs")
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")
ENV = {**os.environ, "KAGGLE_CONFIG_DIR": os.path.expanduser("~/.kaggle")}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd, check=False):
    r = subprocess.run([KAGGLE] + cmd, env=ENV, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(cmd)} failed:\n{out}")
    return r.returncode, out


# ---------- code dataset ----------------------------------------------------
def zip_src(zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(os.path.join(SRC, "causal_maps")):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    z.write(full, os.path.relpath(full, SRC))  # arc: causal_maps/...
                    n += 1
    _log(f"zipped {n} modules -> {zip_path}")


def push_code(msg="update", code_slug=CODE_SLUG):
    slug_name = code_slug.split("/")[-1]
    staging = os.path.join(WORK, f"code-{slug_name}")
    os.makedirs(staging, exist_ok=True)
    zip_src(os.path.join(staging, "causal_maps_code.zip"))
    with open(os.path.join(staging, "dataset-metadata.json"), "w") as f:
        json.dump({"title": slug_name, "id": code_slug,
                   "licenses": [{"name": "CC0-1.0"}]}, f)
    rc, out = _run(["datasets", "version", "-p", staging, "-m", msg])
    if rc != 0:
        _log("version failed (dataset likely absent) -> attempting create")
        rc2, out2 = _run(["datasets", "create", "-p", staging])
        if rc2 != 0:
            raise RuntimeError(f"dataset version AND create failed:\n"
                               f"VERSION:\n{out}\nCREATE:\n{out2}")
        out = out2
    _log(f"code dataset pushed: {out.strip().splitlines()[-1] if out.strip() else 'ok'}")
    wait_dataset_ready(code_slug)  # Kaggle must finish processing before a kernel can mount it


def wait_dataset_ready(slug, timeout=240, poll=5, grace=75):
    """Poll until the dataset is processed, then wait `grace` seconds more.
    `datasets status` is VERSION-BLIND: right after pushing a new version it
    reports the PREVIOUS version's 'ready', and a kernel launched immediately
    mounts stale code (bit us twice: empty mount 2026-07-12; stale stage
    KeyError 2026-07-14). The grace period lets the new version finish
    processing before any kernel push."""
    t0 = time.time()
    ok = False
    while time.time() - t0 <= timeout:
        _rc, out = _run(["datasets", "status", slug])
        if "ready" in out.strip().lower():
            ok = True
            break
        time.sleep(poll)
    if not ok:
        _log(f"WARN dataset {slug} not 'ready' after {timeout}s; proceeding anyway")
    if grace:
        _log(f"dataset {slug} ready; holding {grace}s for version processing")
        time.sleep(grace)
    return ok


# ---------- kernels ---------------------------------------------------------
SMOKE_CODE = '''
import time, platform, os, sys
print("SMOKE start", time.strftime("%H:%M:%S"), flush=True)
print("python", platform.python_version(), flush=True)
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "no CUDA in kernel!"
print("gpu", torch.cuda.get_device_name(0),
      "mem_GB", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), flush=True)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", float((x @ x).float().sum()), flush=True)
os.makedirs("/kaggle/working", exist_ok=True)
open("/kaggle/working/smoke_ok.txt", "w").write("ok\\n")
print("SMOKE done", flush=True)
'''

STAGE_TMPL = '''
import os, sys, json, glob, zipfile
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

inp = "/kaggle/input"
print("KAGGLE INPUT MOUNTS:", os.listdir(inp) if os.path.isdir(inp) else "NONE", flush=True)

# A Kaggle dataset can retain files deleted in later versions. Prefer the
# versioned zip explicitly: an older unpacked causal_maps directory may coexist
# with it and must not shadow the current source revision.
zips = glob.glob(inp + "/**/causal_maps_code.zip", recursive=True)
if zips:
    newest = max(zips, key=os.path.getmtime)
    zipfile.ZipFile(newest).extractall("/tmp/code"); sys.path.insert(0, "/tmp/code")
    print("extracted versioned code", newest, flush=True)
else:
    hits = glob.glob(inp + "/**/causal_maps/__init__.py", recursive=True)
    if hits:
        parent = os.path.dirname(os.path.dirname(hits[0]))
        sys.path.insert(0, parent)
        print("using package fallback at", parent, flush=True)
    else:
        print("FULL INPUT TREE:", flush=True)
        for p in sorted(glob.glob(inp + "/**", recursive=True))[:100]:
            print("  ", p, flush=True)
        raise SystemExit("causal_maps not found under /kaggle/input")

STAGE = {stage!r}
CONFIG = json.loads({config_json!r})

# Transformers 5 delegates AWQ execution to gptqmodel. Install it before any
# causal_maps import (model_utils imports transformers at module scope), so pip
# cannot leave old and upgraded Transformers modules mixed in one process.
if CONFIG.get("quantization") == "awq":
    import importlib.metadata, subprocess
    try:
        _gptq_ver = importlib.metadata.version("gptqmodel")
    except importlib.metadata.PackageNotFoundError:
        _gptq_ver = None
    if _gptq_ver != "7.1.0":
        print("installing gptqmodel==7.1.0 before transformers import",
              flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "gptqmodel==7.1.0"], check=True)

from causal_maps import experiments
experiments.main(STAGE, CONFIG, out_dir="/kaggle/working")
'''


def write_kernel(work, slug, title, code, dataset_sources=(), model_sources=(),
                 accelerator="NvidiaTeslaT4", internet=True):
    """accelerator (machine_shape): 'NvidiaTeslaT4' | 'NvidiaTeslaP100' | None
    (CPU). enable_gpu alone is legacy and NOT honored server-side — machine_shape
    is what actually attaches a GPU."""
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "kernel_main.py"), "w") as f:
        f.write(code)
    meta = {
        "id": slug, "title": title, "code_file": "kernel_main.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true" if accelerator else "false", "enable_tpu": "false",
        "enable_internet": "true" if internet else "false",
        "dataset_sources": list(dataset_sources),
        "model_sources": list(model_sources),
        "competition_sources": [], "kernel_sources": [],
    }
    if accelerator:
        meta["machine_shape"] = accelerator
    with open(os.path.join(work, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    _log(f"kernel metadata -> {slug} (accelerator={accelerator}, internet={internet}, "
         f"datasets={list(dataset_sources)})")


def push_kernel(work, accelerator=None):
    cmd = ["kernels", "push", "-p", work]
    if accelerator:  # pass explicitly too (CLI acc overrides metadata machine_shape)
        cmd += ["--accelerator", accelerator]
    rc, out = _run(cmd, check=True)
    _log(f"pushed: {out.strip().splitlines()[-1] if out.strip() else 'ok'}")


def kernel_status(slug):
    """Return a coarse status keyword. Scans the raw output for known states
    (robust to plain 'complete' or enum-style 'KernelWorkerStatus.COMPLETE')."""
    rc, out = _run(["kernels", "status", slug])
    low = out.lower()
    for key in ("complete", "error", "cancelacknowledged", "cancelrequested",
                "running", "queued"):
        if key in low:
            return key, out.strip()
    return "unknown", out.strip()


def wait(slug, max_wait=3600, poll=30):
    t0 = time.time()
    warned = False
    while True:
        st, raw = kernel_status(slug)
        el = time.time() - t0
        _log(f"status={st} elapsed={el:.0f}s")
        if st in ("complete", "error", "cancelacknowledged", "cancelrequested"):
            return st
        if el > max_wait and not warned:
            _log(f"WARN exceeded max_wait={max_wait}s and still {st} — possible stall "
                 f"(will keep polling to 2x)")
            warned = True
        if el > 2 * max_wait:
            _log("WARN hard cap hit; stopping poll (kernel may still be running)")
            return st
        time.sleep(poll)


def pull_output(slug, dest):
    os.makedirs(dest, exist_ok=True)
    rc, out = _run(["kernels", "output", slug, "-p", dest])
    _log(f"pulled output -> {dest}")
    # show log tail + results if present
    for fn in sorted(os.listdir(dest)):
        if fn.endswith(".log"):
            with open(os.path.join(dest, fn)) as f:
                lines = f.read().splitlines()
            _log(f"--- {fn} tail (last 25) ---")
            for ln in lines[-25:]:
                print("   " + ln, flush=True)
    for fn in sorted(os.listdir(dest)):
        if fn.startswith("results_") and fn.endswith(".json"):
            with open(os.path.join(dest, fn)) as f:
                _log(f"--- {fn} ---")
                print(json.dumps(json.load(f), indent=2), flush=True)


# ---------- top-level flows -------------------------------------------------
def do_smoke(max_wait, poll, accelerator="NvidiaTeslaT4"):
    slug = f"{USER}/cm-smoke2"  # fresh slug (cm-smoke was created CPU-only pre-fix)
    work = os.path.join(WORK, "smoke")
    write_kernel(work, slug, "cm-smoke2", SMOKE_CODE, accelerator=accelerator, internet=False)
    push_kernel(work, accelerator=accelerator)
    st = wait(slug, max_wait, poll)
    pull_output(slug, os.path.join(RUNS, "smoke"))
    _log(f"SMOKE final status: {st}")
    return st


def do_run(stage, config, max_wait, poll, refresh_code=True,
           accelerator="NvidiaTeslaT4", slug_suffix=None,
           code_slug=CODE_SLUG, dataset_sources=(), internet=True,
           kernel_slug=None, model_sources=()):
    if refresh_code:
        push_code(msg=f"for {stage}", code_slug=code_slug)
    suffix = f"-{slug_suffix}" if slug_suffix else ""
    slug = f"{USER}/{kernel_slug}" if kernel_slug else (
        f"{USER}/cm-{stage.replace('_', '-')}{suffix}")
    work = os.path.join(WORK, f"{stage}{suffix}")
    code = STAGE_TMPL.format(stage=stage, config_json=json.dumps(config))
    title = kernel_slug if kernel_slug else f"cm-{stage}{suffix}"
    write_kernel(work, slug, title, code,
                 dataset_sources=[code_slug, *dataset_sources],
                 model_sources=list(model_sources),
                 accelerator=accelerator, internet=internet)
    push_kernel(work, accelerator=accelerator)
    st = wait(slug, max_wait, poll)
    pull_output(slug, os.path.join(RUNS, f"{stage}{suffix}"))
    _log(f"{stage} final status: {st}")
    return st


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push-code")
    sp = sub.add_parser("smoke")
    sp.add_argument("--max-wait", type=int, default=900)
    sp.add_argument("--poll", type=int, default=20)
    sp.add_argument("--accelerator", default="NvidiaTeslaT4")
    rp = sub.add_parser("run")
    rp.add_argument("stage")
    rp.add_argument("--config", default="{}", help="JSON config for experiments.main")
    rp.add_argument("--max-wait", type=int, default=3600)
    rp.add_argument("--poll", type=int, default=30)
    rp.add_argument("--accelerator", default="NvidiaTeslaT4")
    rp.add_argument("--no-refresh-code", action="store_true")
    rp.add_argument("--slug-suffix", help="append a distinct suffix for comparable model runs")
    rp.add_argument("--code-dataset", default=CODE_SLUG,
                    help="code dataset slug; use a new slug for immutable research runs")
    rp.add_argument("--dataset-source", action="append", default=[],
                    help="additional mounted Kaggle dataset (repeatable)")
    rp.add_argument("--model-source", action="append", default=[],
                    help="mounted Kaggle model ref owner/model/framework/variation/version (repeatable)")
    rp.add_argument("--no-internet", action="store_true")
    rp.add_argument("--kernel-slug",
                    help="short Kaggle kernel id without the username prefix")
    stp = sub.add_parser("status"); stp.add_argument("slug")
    op = sub.add_parser("output"); op.add_argument("slug"); op.add_argument("--dest", default=RUNS)
    a = ap.parse_args()

    if a.cmd == "push-code":
        push_code()
    elif a.cmd == "smoke":
        do_smoke(a.max_wait, a.poll, accelerator=a.accelerator)
    elif a.cmd == "run":
        do_run(a.stage, json.loads(a.config), a.max_wait, a.poll,
               refresh_code=not a.no_refresh_code, accelerator=a.accelerator,
               slug_suffix=a.slug_suffix, code_slug=a.code_dataset,
               dataset_sources=a.dataset_source, internet=not a.no_internet,
               kernel_slug=a.kernel_slug, model_sources=a.model_source)
    elif a.cmd == "status":
        st, raw = kernel_status(a.slug); _log(f"{a.slug}: {st}\n{raw}")
    elif a.cmd == "output":
        pull_output(a.slug, os.path.join(a.dest, a.slug.split('/')[-1]))


if __name__ == "__main__":
    main()
