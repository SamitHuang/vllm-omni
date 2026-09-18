# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph capture for Qwen-Image-2.1 KV-cache decode steps.

Decode (denoising steps 1..N) recomputes only the target image's tokens: the
shape is identical every step and the timestep-independent prefix K/V come from
the cross-step KV cache. That is the textbook CUDA-graph case, except the
default cache stores prefix K/V in freshly allocated tensors, so a captured
graph would bind addresses that the next request (or a step-mode cache merge)
no longer owns.

This module keeps the prefix K/V in persistent, fixed-address buffers. Prefill
copies its K/V into them and re-points the caller's cache dict at the buffers;
decode then replays a captured graph that reads those buffers plus static input
buffers (target latents, timestep, RoPE frequencies). One graph is captured per
``(branch, batch, prefix_len, target_tokens, dtype, device, backend)`` key and
reused across requests: a new prefill with the same key rewrites the buffers in
place, which a captured graph reads correctly because it binds addresses, not
contents.

Memory note: only the *prefix* K/V are persistent. The decode attention still
concatenates prefix K/V with the current step's target K/V inside the captured
region, and that transient tensor is served by the graph's memory pool, so the
persistent footprint is the same as the existing dict cache.

Fallbacks (all logged, all eager): CUDA unavailable, sequence/ring/tensor
parallelism, HSDP/offload hooks, torch.compile'd blocks, KV-cache quantization,
dynamic LoRA wrappers, padded text masks (the masked attention path branches on
mask contents, which cannot be captured), unknown graph key at decode, and
capture failure.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Transformer2DModel,
    )

logger = init_logger(__name__)


def _dynamic_lora_wrappers_present(module: torch.nn.Module) -> bool:
    """True once ``DiffusionLoRAManager`` has wrapped layers under ``module``.

    A captured graph records the layers and branches that ran at capture time;
    binding or rescaling an adapter afterwards changes the math without
    changing the shapes the graph was keyed on, so graphs and dynamic LoRA
    must not mix. Same reasoning as ``sensenova_u1.paged_decode``.
    """
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
    except ImportError:  # pragma: no cover - depends on the wheel
        return False
    return any(isinstance(m, BaseLayerWithLoRA) for m in module.modules())


class QwenImage21DecodeGraphEntry:
    """Static state for one captured decode graph.

    Buffers are created outside inference mode so replay-side ``copy_`` is
    legal regardless of the caller's grad context. ``graph`` is None until the
    first decode step captures it; ``failed`` disables capture permanently for
    this key after an exception.
    """

    def __init__(
        self,
        *,
        branch: str,
        batch_size: int,
        prefix_len: int,
        target_tokens: int,
        in_channels: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_shapes: list[torch.Size],
        kv_dtype: torch.dtype,
        freqs: torch.Tensor,
    ):
        self.branch = branch
        self.target_tokens = target_tokens
        self.prefix_len = prefix_len
        with torch.inference_mode(False):
            self.hidden = torch.zeros(batch_size, target_tokens, in_channels, dtype=dtype, device=device)
            self.timestep = torch.zeros(batch_size, dtype=dtype, device=device)
            self.k = [torch.zeros(shape, dtype=kv_dtype, device=device) for shape in kv_shapes]
            self.v = [torch.zeros(shape, dtype=kv_dtype, device=device) for shape in kv_shapes]
            self.target_token_mask = torch.ones(target_tokens, dtype=torch.bool, device=device)
            # RoPE frequencies are layout-dependent but step-independent. Clone
            # into a normal (non-inference) tensor: the prefill runs under
            # inference_mode, and the captured body takes views of this tensor
            # outside it, which is forbidden for inference tensors.
            self.freqs = freqs.clone()
        # Decode needs no mask once padding is excluded (see the manager's
        # prefill gate), so the captured body attends without metadata.
        self.attn_metadata = None
        # Per-block cache dicts with the same shape as the legacy protocol.
        self.block_caches = [{branch: {"key": self.k[i], "value": self.v[i]}} for i in range(len(kv_shapes))]
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None
        self.failed = False
        self.captures = 0

    def capture(self, body) -> None:
        """Warm up on a side stream, then capture one decode step.

        Warm-up runs the same computation it will record, allocating cuBLAS
        workspaces and any lazy buffers outside the capture. Decode writes
        nothing to the cache (prefix K/V are read-only), so warm-up is
        idempotent.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode(False), torch.no_grad():
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        # The platform-wide pool is shared with every other captured path in
        # the tree; entries only keep their *output* alive, whose tiny size
        # makes reuse safe because the caller clones it before any other graph
        # on the pool replays.
        with torch.inference_mode(False), torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                self.output = body()
        self.graph = graph
        self.captures += 1


class QwenImage21DecodeGraphManager:
    """Owns the static KV buffers and captured graphs of one transformer."""

    def __init__(self, model: QwenImage21Transformer2DModel, max_entries: int = 8):
        self.model = model
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple, QwenImage21DecodeGraphEntry] = OrderedDict()
        self._static_eligible: bool | None = None

    # ── eligibility ──

    def _check_static_eligibility(self) -> bool:
        """One-time checks that cannot change after weights are loaded."""
        model = self.model
        reason = None
        param = next(model.parameters(), None)
        if param is None or param.device.type != "cuda":
            reason = f"model is not on CUDA (device={None if param is None else param.device})"
        else:
            parallel_config = getattr(model, "parallel_config", None)
            if parallel_config is not None:
                sp = getattr(parallel_config, "sequence_parallel_size", None) or 1
                if sp > 1 or getattr(parallel_config, "ring_degree", 1) > 1:
                    reason = "sequence/ring parallelism changes decode shapes per rank"
                elif getattr(parallel_config, "use_hsdp", False):
                    reason = "HSDP wraps blocks with gather hooks outside the captured region"
            if reason is None:
                from vllm.distributed import get_tensor_model_parallel_world_size

                if get_tensor_model_parallel_world_size() > 1:
                    reason = "tensor parallelism runs collectives inside the captured region"
            if reason is None:
                for block in model.transformer_blocks:
                    if getattr(block.forward, "_torchdynamo_orig_callable", None) is not None:
                        reason = "transformer blocks are torch.compile'd"
                        break
                    if hasattr(block, "_hook_registry") or hasattr(block, "_omni_original_forward"):
                        reason = "transformer blocks carry offload/cache hooks"
                        break
            if reason is None and model.transformer_blocks:
                attn_layer = model.transformer_blocks[0].attn.attn
                if getattr(attn_layer, "_kv_cache_dtype", None) is not None:
                    reason = "KV-cache quantization is enabled"

        if reason is not None:
            logger.warning_once("Qwen-Image-2.1 CUDA graph decode disabled: %s. Falling back to eager decode.", reason)
            return False
        return True

    def eligible(self) -> bool:
        if self._static_eligible is None:
            self._static_eligible = self._check_static_eligibility()
        return self._static_eligible

    def _backend_name(self) -> str:
        backend = self.model.transformer_blocks[0].attn.attn.attn_backend
        return backend.get_name() if backend is not None else "custom"

    # ── prefill registration ──

    def register_prefill(
        self,
        *,
        kv_cache: list[dict[str, dict[str, torch.Tensor]]],
        cache_branch: str,
        prefix_len: int,
        target_tokens: int,
        target_freqs: torch.Tensor,
        joint_key_valid: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        """Move a freshly prefilled prefix K/V into static buffers.

        The caller's ``kv_cache`` dicts are re-pointed at the static buffers, so
        the existing decode path reads fixed addresses and a captured graph can
        be replayed for later steps. Prefill itself always runs eagerly.
        """
        if not self.eligible() or torch.compiler.is_compiling():
            return
        if _dynamic_lora_wrappers_present(self.model):
            if self.entries:
                logger.warning_once(
                    "Qwen-Image-2.1 CUDA graph decode: LoRA wrappers appeared; "
                    "dropping captured graphs and falling back to eager decode."
                )
                self.entries.clear()
            return
        if joint_key_valid is not None and not bool(joint_key_valid.all()):
            # The masked decode path (varlen-unpack or 4D mask) branches on mask
            # contents and cannot be captured; this request stays eager.
            logger.warning_once(
                "Qwen-Image-2.1 CUDA graph decode: padded text mask present; this request falls back to eager decode."
            )
            return

        first = kv_cache[0].get(cache_branch)
        if first is None or "key" not in first:
            return
        batch_size = first["key"].shape[0]
        key = (
            cache_branch,
            batch_size,
            prefix_len,
            target_tokens,
            dtype,
            first["key"].device.index,
            self._backend_name(),
        )

        entry = self.entries.get(key)
        if entry is None:
            while len(self.entries) >= self.max_entries:
                evicted_key, _ = self.entries.popitem(last=False)
                logger.debug("Evicting decode graph entry %s (max_entries=%d)", evicted_key, self.max_entries)
            entry = QwenImage21DecodeGraphEntry(
                branch=cache_branch,
                batch_size=batch_size,
                prefix_len=prefix_len,
                target_tokens=target_tokens,
                in_channels=self.model.in_channels,
                dtype=dtype,
                device=first["key"].device,
                kv_shapes=[block_cache[cache_branch]["key"].shape for block_cache in kv_cache],
                kv_dtype=first["key"].dtype,
                freqs=target_freqs,
            )
            self.entries[key] = entry
        else:
            self.entries.move_to_end(key)

        for i, block_cache in enumerate(kv_cache):
            parts = block_cache[cache_branch]
            entry.k[i].copy_(parts["key"])
            entry.v[i].copy_(parts["value"])
            parts["key"] = entry.k[i]
            parts["value"] = entry.v[i]
        logger.debug(
            "Registered decode graph entry key=%s (entries=%d, captured=%d)",
            key,
            len(self.entries),
            sum(1 for e in self.entries.values() if e.graph is not None),
        )

    # ── decode replay ──

    def try_decode(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: list[dict[str, dict[str, torch.Tensor]]],
        cache_branch: str,
        target_tokens: int,
    ) -> torch.Tensor | None:
        """Replay the captured decode step, or return None to fall back to eager."""
        if not self.eligible() or torch.compiler.is_compiling():
            return None
        first = kv_cache[0][cache_branch]
        cached_key = first["key"]
        if cached_key.device.type != "cuda":
            # Registration is device-agnostic (unit tests exercise the buffer
            # machinery on CPU), but capture/replay is CUDA-only: an empty
            # graph over CPU ops would replay as a stale no-op.
            return None
        key = (
            cache_branch,
            cached_key.shape[0],
            cached_key.shape[1],
            target_tokens,
            hidden_states.dtype,
            cached_key.device.index,
            self._backend_name(),
        )
        entry = self.entries.get(key)
        if entry is None or entry.failed:
            return None
        self.entries.move_to_end(key)

        entry.hidden.copy_(hidden_states[:, -target_tokens:])
        entry.timestep.copy_(timestep)
        # The caller's cache dicts normally already reference the entry's
        # buffers (prefill re-pointed them). A step-mode batched decode instead
        # hands over freshly merged tensors; copy those into the static buffers.
        for i, block_cache in enumerate(kv_cache):
            parts = block_cache[cache_branch]
            if parts["key"] is not entry.k[i]:
                entry.k[i].copy_(parts["key"])
                entry.v[i].copy_(parts["value"])
                parts["key"] = entry.k[i]
                parts["value"] = entry.v[i]

        if entry.graph is None:
            try:
                entry.capture(lambda: self.model._decode_graph_forward(entry))
            except Exception as exc:
                entry.failed = True
                entry.graph = None
                logger.warning(
                    "Qwen-Image-2.1 CUDA graph capture failed for key=%s: %s. "
                    "Falling back to eager decode for this shape.",
                    key,
                    exc,
                )
                return None
            logger.info("Captured Qwen-Image-2.1 decode graph for key=%s", key)
        entry.graph.replay()
        assert entry.output is not None
        # Clone before returning: the output lives in the shared graph pool and
        # the next replay (of this or another graph on the pool) overwrites it.
        return entry.output.clone()

    def clear(self) -> None:
        """Drop all entries, freeing buffers and captured graphs."""
        self.entries.clear()
