"""Activation patching over (layer, position) on minimal pairs.

CONVENTION (stated once, relied on everywhere):
  We patch the OUTPUT of decoder layer L (the post-layer residual stream) at a
  single sequence position p, overwriting it with the counterfactual run's
  residual stream at (L, p). Causal attention then propagates that change to all
  positions >= p in layers > L; we read the answer logit-difference at the final
  position.

      IE(L, p) = logit_diff(clean input, patched at (L,p)) - logit_diff(clean)

  logit_diff = logit(cf_answer) - logit(clean_answer)  (log-odds between the two
  candidate answer tokens). A POSITIVE IE means patching at (L,p) pushed the
  clean run toward the counterfactual answer => that site carries the causal
  information the counterfactual changed.

Hook robustness: a decoder layer's forward output is a bare tensor in
transformers 5.x and a tuple in 4.x — every hook here handles both.
"""
import os

import numpy as np
import torch

from .logutil import Heartbeat, log
from .model_utils import (get_decoder_layers, input_device, last_token_logits,
                          logit_diff)


def _split_output(out):
    """Return (hidden_state_tensor, rebuild_fn) for a layer forward output that
    may be a bare tensor (transformers 5.x) or a tuple (4.x)."""
    if isinstance(out, tuple):
        rest = out[1:]
        return out[0], (lambda hs: (hs,) + tuple(rest))
    return out, (lambda hs: hs)


@torch.no_grad()
def cache_layer_outputs(model, input_ids, attention_mask, to_cpu=False):
    """One forward pass, capturing every decoder layer's output hidden state.
    Returns a list indexed by layer -> tensor [B, S, D]."""
    layers = get_decoder_layers(model)
    cache = [None] * len(layers)
    handles = []

    def mk(i):
        def hook(module, inp, out):
            hs, _ = _split_output(out)
            cache[i] = hs.detach().to("cpu") if to_cpu else hs.detach().clone()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(mk(i)))
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cache


@torch.no_grad()
def forward_with_patch(model, input_ids, attention_mask, layer_idx, position, value):
    """Forward on `input_ids` with layer `layer_idx`'s output at `position`
    overwritten by `value` ([B, D]). Returns last-token logits [B, V]."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        hs[:, position, :] = value.to(dtype=hs.dtype, device=hs.device)
        return rebuild(hs)

    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]


@torch.no_grad()
def forward_with_add(model, input_ids, attention_mask, layer_idx, position, delta,
                     scale=1.0):
    """Forward with `scale * delta` ADDED to residual at (layer, position).
    `delta` is [D] or [B, D]. Returns last-token logits [B, V]."""
    layer = get_decoder_layers(model)[layer_idx]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        d = delta.to(dtype=hs.dtype, device=hs.device)
        if d.dim() == 1:
            hs[:, position, :] = hs[:, position, :] + scale * d
        else:
            hs[:, position, :] = hs[:, position, :] + scale * d
        return rebuild(hs)

    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]


@torch.no_grad()
def forward_with_project(model, input_ids, attention_mask, layer_idx, position,
                         basis, alpha=1.0):
    """Directional ablation: h ← h − α P_S h at (layer, position).

    `basis` is [D] (1-D) or [D, K] with orthonormal columns spanning S.
    Returns last-token logits [B, V].
    """
    layer = get_decoder_layers(model)[layer_idx]
    U = basis.float()
    if U.dim() == 1:
        U = U / U.norm().clamp(min=1e-8)
        U = U.unsqueeze(1)  # [D, 1]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        h = hs[:, position, :].float()  # [B, D]
        Udev = U.to(dtype=h.dtype, device=h.device)
        # proj = (h @ U) @ U.T
        proj = (h @ Udev) @ Udev.T
        hs[:, position, :] = (h - float(alpha) * proj).to(dtype=hs.dtype)
        return rebuild(hs)

    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]


@torch.no_grad()
def forward_with_mean_knock(model, input_ids, attention_mask, layer_idx, position):
    """Site knockout: replace h[*, position, :] with the batch mean at that position.

    Removes example-specific content at the site while preserving average activation.
    Returns last-token logits [B, V].
    """
    layer = get_decoder_layers(model)[layer_idx]

    def hook(module, inp, out):
        hs, rebuild = _split_output(out)
        hs = hs.clone()
        mean = hs[:, position, :].mean(dim=0, keepdim=True)
        hs[:, position, :] = mean.expand_as(hs[:, position, :])
        return rebuild(hs)

    handle = layer.register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return out.logits[:, -1, :]


def _ckpt_path(out_dir, tag):
    return os.path.join(out_dir, f"ckpt_{tag}.npz") if out_dir else None


@torch.no_grad()
def sweep_ie(model, clean, cf, pos_ids, neg_ids, layers=None, positions=None,
             out_dir=None, tag="sweep", cache_to_cpu=False, hb_every=30.0,
             ckpt_every=200):
    """Full (layer x position) IE sweep over a batch of minimal pairs.

    clean, cf: dicts with input_ids [B,S] and attention_mask [B,S]. Clean and
        counterfactual must share shape S (equal-length design so positions
        align across the batch).
    pos_ids, neg_ids: [B] LongTensors. pos = counterfactual answer id, neg =
        clean answer id (per pair).

    Returns:
        ie_mean [n_layers_swept, n_pos_swept]  (mean over pairs)
        ie_all  [n_layers_swept, n_pos_swept, B]  (per-pair, for split-half/nulls)
        clean_ld [B]  (baseline logit_diff of the clean run)
        meta dict (layers, positions swept)
    Checkpoints ie_all to out_dir/ckpt_<tag>.npz every `ckpt_every` steps and
    resumes from it if present (Kaggle sessions die)."""
    dev = input_device(model)
    ci = clean["input_ids"].to(dev)
    cam = clean["attention_mask"].to(dev)
    B, S = ci.shape
    n_layers = len(get_decoder_layers(model))
    layers = list(range(n_layers)) if layers is None else list(layers)
    positions = list(range(S)) if positions is None else list(positions)
    pos_t = pos_ids.to(dev)
    neg_t = neg_ids.to(dev)

    # Baseline (clean) logit_diff and the counterfactual activation cache.
    clean_ld = logit_diff(last_token_logits(model, ci, cam), pos_t, neg_t)
    clean_ld_np = clean_ld.float().cpu().numpy()
    cf_cache = cache_layer_outputs(model, cf["input_ids"].to(dev),
                                   cf["attention_mask"].to(dev), to_cpu=cache_to_cpu)

    nL, nP = len(layers), len(positions)
    ie = np.full((nL, nP, B), np.nan, dtype=np.float32)

    # Resume?
    start_step = 0
    ckpt = _ckpt_path(out_dir, tag)
    if ckpt and os.path.exists(ckpt):
        d = np.load(ckpt)
        if d["ie"].shape == ie.shape:
            ie = d["ie"]
            start_step = int(d["step"][0])
            log(f"resumed {tag} from step {start_step}/{nL*nP}")

    hb = Heartbeat(total=nL * nP, stage=tag, every_sec=hb_every, out_dir=out_dir)
    step = 0
    for li, L in enumerate(layers):
        for pi, P in enumerate(positions):
            step += 1
            if step <= start_step:
                continue
            value = cf_cache[L][:, P, :]  # [B, D]
            patched = forward_with_patch(model, ci, cam, L, P, value)
            ld = logit_diff(patched, pos_t, neg_t)
            ie[li, pi, :] = (ld - clean_ld).float().cpu().numpy()
            hb.step(step, extra=f"L={L} P={P} meanIE={np.nanmean(ie[li, pi]):+.3f}")
            if ckpt and (step % ckpt_every == 0):
                np.savez(ckpt, ie=ie, step=np.array([step]))
    if ckpt:
        np.savez(ckpt, ie=ie, step=np.array([nL * nP]))
    hb.done()

    ie_mean = np.nanmean(ie, axis=2)
    meta = {"layers": layers, "positions": positions, "B": B, "S": S}
    return ie_mean, ie, clean_ld_np, meta
