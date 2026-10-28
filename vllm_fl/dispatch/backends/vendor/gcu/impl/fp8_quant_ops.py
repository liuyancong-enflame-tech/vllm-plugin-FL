# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU implementations for the ``_C`` FP8 activation-quantization ops.

``vllm/_custom_ops.py::scaled_fp8_quant`` dispatches straight into three
``_C`` ops that are CUDA-only upstream:

    static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale,
                            (int, int)? group_shape=None) -> ()
    dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) -> ()
    dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input,
                                       Tensor! scale, Tensor? scale_ub) -> ()

``vllm_fl`` registers their *schemas* (``vllm_fl/ops/_C_ops_schemas.py``) so
torch.compile pattern matching works, but no implementation.  Without one, any
non-block FP8 checkpoint dies:

* while loading weights -- ``process_fp8_weight_tensor_strategy`` ->
  ``requantize_with_max_scale`` calls ``ops.scaled_fp8_quant`` on **CPU**
  tensors;
* at runtime -- static / per-token / per-tensor activation quantization.

This module registers GCU-only ``CompositeExplicitAutograd`` implementations
(the shared registry stays untouched, so other OOT vendors keep the previous
"schema only" behaviour).  The math follows ``QuantFP8.forward_native``.

Note the elementwise chain below is written with plain aten ops on purpose: on
GCU, ``flag_gems.enable()`` routes ``reciprocal`` / ``mul`` / ``clamp`` to
FlagGems Triton kernels, while the final ``.to(float8_e4m3fn)`` is done by
topsaten.  A single fused Triton kernel is *not* possible today because
triton_gcu400 mis-lowers scalar f32 -> fp8 casts (``llvm.fptrunc`` with an i8
result); see ``repro_triton_gcu_fp8_truncf.py``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False

_IMPL_NAMES = (
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
)


def _fp8_consts():
    from vllm.model_executor.layers.quantization.input_quant_fp8 import (
        _FP8_MAX,
        _FP8_MIN,
        _FP8_MIN_SCALING_FACTOR,
    )

    return _FP8_MAX, _FP8_MIN, _FP8_MIN_SCALING_FACTOR


def _broadcast_scale(scale: torch.Tensor, x: torch.Tensor, group_shape) -> torch.Tensor:
    """Expand a checkpoint scale to the elementwise shape of ``x``.

    ``group_shape == (gm, gn)`` with ``-1`` meaning "whole extent"; ``None``
    means per-tensor scaling.
    """
    s = scale.to(torch.float32)
    if group_shape is None or s.numel() == 1:
        return s.reshape(()) if s.numel() == 1 else s

    rows, cols = x.shape[-2], x.shape[-1]
    gm, gn = int(group_shape[0]), int(group_shape[1])
    if gm in (-1, 0):
        gm = rows
    if gn in (-1, 0):
        gn = cols

    blocks_m = (rows + gm - 1) // gm
    blocks_n = (cols + gn - 1) // gn
    s = s.reshape(-1)
    if s.numel() == rows:  # per-channel / per-token along rows
        return s.view(-1, 1)
    if s.numel() == cols:  # per-channel along columns
        return s.view(1, -1)
    if s.numel() < blocks_m * blocks_n:
        s = s[: blocks_m * blocks_n]
    s = s.view(blocks_m, blocks_n)
    s = s.repeat_interleave(gm, dim=0)[:rows]
    return s.repeat_interleave(gn, dim=1)[:, :cols]


def _quant_from_scale(x: torch.Tensor, s: torch.Tensor, out_dtype) -> torch.Tensor:
    fp8_max, fp8_min, _ = _fp8_consts()
    out = (x.to(torch.float32) * s.to(torch.float32).reciprocal()).clamp(fp8_min, fp8_max)
    return out.to(out_dtype)


def static_scaled_fp8_quant(
    result, input, scale, group_shape=None
) -> None:
    """``result = clamp(input / scale)`` with a checkpoint-provided scale."""
    s = _broadcast_scale(scale, input, group_shape)
    result.copy_(_quant_from_scale(input, s, result.dtype))


def dynamic_scaled_fp8_quant(result, input, scale) -> None:
    """Per-tensor dynamic scaling; writes the scale back into ``scale``."""
    fp8_max, _, min_scale = _fp8_consts()
    amax = input.to(torch.float32).abs().max().reshape(1)
    s = (amax / fp8_max).clamp(min=min_scale)
    scale.copy_(s.to(scale.dtype))
    result.copy_(_quant_from_scale(input, s, result.dtype))


def dynamic_per_token_scaled_fp8_quant(result, input, scale, scale_ub=None) -> None:
    """Per-row (per-token) dynamic scaling with an optional upper bound."""
    fp8_max, _, min_scale = _fp8_consts()
    x = input.to(torch.float32)
    amax = x.abs().max(dim=-1, keepdim=True).values
    if scale_ub is not None:
        amax = amax.clamp(max=float(scale_ub.max()))
    s = (amax / fp8_max).clamp(min=min_scale)
    scale.copy_(s.reshape(scale.shape).to(scale.dtype))
    result.copy_(_quant_from_scale(x, s, result.dtype))


_GCU_IMPLS = (
    ("static_scaled_fp8_quant", static_scaled_fp8_quant),
    ("dynamic_scaled_fp8_quant", dynamic_scaled_fp8_quant),
    ("dynamic_per_token_scaled_fp8_quant", dynamic_per_token_scaled_fp8_quant),
)


def _is_gcu_platform() -> bool:
    try:
        from vllm.platforms import current_platform

        return current_platform.device_type == "gcu"
    except Exception:
        return False


def apply_fp8_quant_ops_gcu_patch() -> None:
    """Register the three ``_C`` fp8 quantization ops for GCU only."""
    global _patched
    if _patched:
        return

    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available() or not _is_gcu_platform():
        return

    try:
        from vllm_fl.ops._C_ops_registry import (
            register_composite_impls,
            register_op_schemas,
        )

        # Idempotent: no-op when vLLM's native extensions are available.
        register_op_schemas()
        registered = register_composite_impls(_GCU_IMPLS)
        if registered:
            _patched = True
            logger.info(
                "Registered GCU implementations for _C fp8 quantization ops: %s",
                ", ".join(n for n, _ in _GCU_IMPLS if n in _IMPL_NAMES),
            )
        else:
            logger.debug("fp8 quant ops patch deferred (schema library not ready)")
    except Exception as exc:
        logger.debug("fp8 quant ops patch deferred (will retry): %s", exc)


__all__ = [
    "apply_fp8_quant_ops_gcu_patch",
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
]
