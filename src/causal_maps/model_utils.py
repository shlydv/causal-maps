"""Model loading, tokenizer helpers, single-token validation, log-odds metric.

Targets Qwen2.5-{1.5B,7B}-Instruct via HF transformers 5.x. Designed for GPU
(Kaggle). bf16, greedy, use_cache=False for the single-forward patching passes.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .logutil import log


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _ensure_bitsandbytes():
    import importlib
    import importlib.metadata
    import subprocess
    import sys
    required = "0.49.2"
    try:
        installed = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != required:
        log(f"installing bitsandbytes=={required} (found {installed})...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--upgrade", f"bitsandbytes=={required}"],
                       check=True)
    importlib.import_module("bitsandbytes")


def _ensure_gptqmodel():
    """Install the Transformers-5 AWQ runtime before checkpoint loading."""
    import importlib
    import importlib.metadata
    import subprocess
    import sys
    required = "7.1.0"
    try:
        installed = importlib.metadata.version("gptqmodel")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != required:
        log(f"installing gptqmodel=={required} (found {installed})...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        f"gptqmodel=={required}"], check=True)
    importlib.import_module("gptqmodel")


def _bnb_config_kwargs(quantization, dtype):
    """Return the frozen bitsandbytes recipe without importing bitsandbytes.

    Kept as a pure helper so the memory-feasibility path can be tested offline.
    """
    if quantization == "8bit":
        return {"load_in_8bit": True}
    if quantization == "4bit":
        return {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": dtype,
            "bnb_4bit_use_double_quant": False,
        }
    raise ValueError(f"unsupported quantization: {quantization!r}")


def _load_dtype_and_map(quantization, dtype, device_map):
    """Resolve checkpoint-native loading choices without touching the model."""
    if quantization == "awq":
        # The official Qwen AWQ checkpoint already stores quantized weights.
        # Its Kaggle recipe requires dtype='auto'; adding a second quantizer
        # would recreate the memory problem this path is meant to avoid.
        return "auto", device_map or "auto"
    return dtype, device_map


def load_model_and_tokenizer(model_path, device=None, dtype=torch.bfloat16,
                             device_map=None, quantization=None, revision=None,
                             trust_remote_code=False, max_memory=None):
    """Load model + tokenizer.
    - quantization='8bit': LLM.int8 via bitsandbytes (~half the memory, so 7B
      fits FULLY on one 16 GB GPU -> no CPU-offloaded layer -> ~10x faster than
      device_map=auto offload). Weights are int8; the residual stream we patch
      stays fp16, so patching fidelity is preserved.
    - quantization='4bit': NF4 weights with bf16 computation. This is the
      memory-safe path for 14B on Kaggle's 2xT4 worker.
    - quantization='awq': load an already-AWQ-quantized checkpoint directly;
      no bitsandbytes conversion or full-weight materialization.
    - device_map set (e.g. 'auto'): accelerate dispatch (may offload to CPU).
    - else: whole model on one device."""
    log(f"loading tokenizer: {model_path}")
    tok = AutoTokenizer.from_pretrained(
        model_path, revision=revision, trust_remote_code=trust_remote_code)
    load_dtype, resolved_map = _load_dtype_and_map(
        quantization, dtype, device_map)
    kw = {}
    if revision is not None:
        kw["revision"] = revision
    if trust_remote_code:
        kw["trust_remote_code"] = True
    if quantization in ("8bit", "4bit"):
        _ensure_bitsandbytes()
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            **_bnb_config_kwargs(quantization, dtype))
        kw["device_map"] = device_map or "auto"
    elif quantization == "awq":
        _ensure_gptqmodel()
        kw["device_map"] = resolved_map
    elif quantization is not None:
        raise ValueError(f"unsupported quantization: {quantization!r}")
    elif device_map is not None:
        kw["device_map"] = device_map
    if max_memory and kw.get("device_map"):
        # JSON configs carry string keys ("0": "11GiB"); accelerate wants ints
        # for GPU ids. Forces balanced sharding (auto packed 14B onto one T4).
        kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v
                            for k, v in max_memory.items()}
    log(f"loading model: {model_path} dtype={load_dtype} "
        f"device_map={kw.get('device_map')} quant={quantization}")
    # transformers 5.x uses `dtype=`; older used `torch_dtype=`. Support both.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=load_dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=load_dtype, **kw)
    if quantization is None and device_map is None:
        model.to(device or pick_device())
    model.eval()
    log(f"loaded: layers={model.config.num_hidden_layers} "
        f"hidden={model.config.hidden_size} input_device={input_device(model)}")
    return model, tok


def input_device(model):
    """Device the input_ids should live on (the embedding's device). Correct for
    both single-device and device_map='auto' (sharded) models."""
    return model.get_input_embeddings().weight.device


def get_decoder_layers(model):
    """The ModuleList of decoder layers (Qwen2/LLaMA-style: model.model.layers).
    Each layer's forward output is the post-layer residual stream; in
    transformers 5.x it is returned as a bare tensor (older: a tuple)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("decoder layers not found (expected model.model.layers)")


def single_token_id(tokenizer, word, leading_space=True):
    """Token id for `word` as it appears mid-sentence (leading space by default).
    Raises if it is not exactly one token — the harness must never track a
    multi-token answer as if it were single (a prior-arc trap)."""
    s = (" " + word) if leading_space else word
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{s!r} -> {ids} is not single-token under this tokenizer")
    return ids[0]


def validate_single_token(tokenizer, words, leading_space=True):
    """Return {word: id} for single-token words; log (and drop) the rest."""
    ok, bad = {}, []
    for w in words:
        try:
            ok[w] = single_token_id(tokenizer, w, leading_space)
        except ValueError:
            bad.append(w)
    if bad:
        log(f"NON-single-token, excluded ({len(bad)}): {bad}")
    return ok


@torch.no_grad()
def last_token_logits(model, input_ids, attention_mask=None):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits[:, -1, :]  # [B, V]


def logit_diff(logits_last, pos_id, neg_id):
    """log-odds between two answer tokens at the measurement position:
        logit(pos) - logit(neg)  ==  log p(pos) - log p(neg)
    (the softmax denominator cancels exactly). `pos`/`neg` may be int or a
    [B] LongTensor of per-example ids. Returns [B]. By convention pos = the
    counterfactual answer, neg = the clean answer, so a positive value means the
    model favours the counterfactual answer."""
    def gather(ids):
        if torch.is_tensor(ids):
            return logits_last.gather(1, ids.to(logits_last.device).view(-1, 1)).squeeze(1)
        return logits_last[:, ids]
    return gather(pos_id) - gather(neg_id)
