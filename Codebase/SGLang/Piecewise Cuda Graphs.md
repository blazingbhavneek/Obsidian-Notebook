python/sglang/srt/model_executor/model_runner.py

\ ## double hash means these comments are added by me, so take care accuracy will be less than original authors comments

```python

class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def can_run_piecewise_cuda_graph(self):
        if self.server_args.enable_torch_compile:
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph has conflict with torch compile",
            )
            return False
            ### Piecewise CUDA graph uses its OWN torch.compile integration.
            ### If user enables torch.compile separately, there's a conflict.
        if self.pp_size > 1:
            # TODO(yuwei): support PP
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because piecewise_cuda_graph does not support PP",
            )
            return False
            ### Pipeline parallelism splits the model across GPUs sequentially.
            ### Piecewise graphs would need to handle cross-GPU boundaries which is complex.
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            # TODO(yuwei): fix the compilation errors for MOE A2A backend
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph due to existing compilation errors",
            )
            return False
        return True


    def init_piecewise_cuda_graphs(self):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if (
            not self.server_args.enable_piecewise_cuda_graph
            or not self.can_run_piecewise_cuda_graph()
        ):
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        self.attention_layers = []
        self.moe_layers = []
        for layer in self.model.model.layers:
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    self.attention_layers.append(layer.self_attn.attn)
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    self.attention_layers.append(layer.self_attn.attn_mqa)
            # For hybrid model
            elif hasattr(layer, "attn"):
                self.attention_layers.append(layer.attn)
            elif hasattr(layer, "linear_attn"):
                self.attention_layers.append(layer.linear_attn)
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    self.attention_layers.append(layer.attention.attn)

            moe_block = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
            self.moe_layers.append(moe_block)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            ## This means only models in which All layers have normal attention are supported only
            ### CORRECT but needs clarification: This check ensures ALL transformer layers have
            ### recognizable attention modules. Some architectures have:
            ### - Hybrid attention/SSM layers (like Jamba, Mamba-based models)
            ### - Non-standard attention patterns
            ### - Layers without attention (pure MLP or SSM)
            ### If we can't find attention in every layer, piecewise graphs won't work correctly
            ### because the backend needs to know exactly where to "cut" the graph at attention ops.
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput]:
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if not self.is_generation:
            kwargs["get_embedding"] = True

        if (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        ):
            return self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)
            ### If piecewise graphs are available and the batch qualifies,
            ### use the fast replay path instead of normal forward.

        if not skip_attn_backend_init:
            self.model_runner.attn_backend.init_forward_metadata(forward_batch)
            ### Attention backend needs to prepare metadata (block tables, etc.)
            ### before the forward pass. skip_attn_backend_init is used when
            ### metadata was already initialized (e.g., in replay_prepare).

        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            **kwargs,
        )

```


python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py

```python
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ... [license header unchanged]

"""Run the model with cuda graph and torch.compile."""

### PIECEWISE CUDA GRAPH OVERVIEW:
### Unlike traditional CUDA graphs that capture the entire forward pass,
### "piecewise" CUDA graphs break the computation into segments. This allows:
### 1. Dynamic operations (like attention with variable sequence lengths) to run outside graphs
### 2. Static operations (like MLP layers) to be captured and replayed efficiently
### 3. Better memory reuse across different input sizes
### This is particularly useful for the prefill/extend phase which has variable token counts.

from __future__ import annotations

import bisect
import gc
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Union

import torch
import tqdm

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled, set_compiled
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    enable_piecewise_cuda_graph_compile,
    set_forward_context,
    set_pcg_capture_stream,
)
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.utils import get_available_gpu_memory, is_npu, log_info_on_rank0

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    ### GC freezing is critical during CUDA graph capture because:
    ### 1. Python's GC can trigger at unpredictable times
    ### 2. If GC runs during capture, freed memory addresses get baked into the graph
    ### 3. When the graph replays, those addresses may point to invalid/different data
    ### gc.freeze() moves all existing objects to a "permanent generation" that won't be collected
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    ### This function traverses the model and tells MultiPlatformOp layers
    ### to switch between their torch.compile-compatible and regular implementations
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(model: torch.nn.Module, compiler: str):
    ### Context manager that temporarily puts model in "compile mode"
    ### MultiPlatformOp layers may have different implementations for
    ### eager vs compiled execution (e.g., using different CUDA kernels)
    try:
        if compiler != "eager":
            _to_torch(model, reverse=False, num_tokens=16)
        yield model
    finally:
        _to_torch(model, reverse=True, num_tokens=16)


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None
### The memory pool is a GPU-side allocation that CUDA graphs can share.
### This prevents each graph from needing its own separate memory allocation,
### reducing overall GPU memory fragmentation and usage.


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


def set_torch_compile_config():
    import torch._dynamo.config

    # Resolve torch._dynamo.exc.FailOnRecompileLimitHit
    ### torch.compile caches compiled versions. When input shapes change too often,
    ### it triggers recompilation. Increasing cache limits prevents crashes when
    ### capturing many different batch sizes/token counts.
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


class PiecewiseCudaGraphRunner:
    """A PiecewiseCudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    ## enable_mamba_extra_buffer + Radix cache enabled + no speculative decoding
    ### CORRECT: This checks if Mamba state tracking is needed. Mamba models have
    ### recurrent state that needs to be tracked across tokens. The extra buffer
    ### stores intermediate Mamba states for cache management with radix trees.
    def is_mamba_track_enabled(self):
        return (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and not self.model_runner.server_args.disable_radix_cache
            and self.model_runner.spec_algorithm.is_none()
        )

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        ## Get args from parent
        ### CORRECT: model_runner is the parent that holds model config, server args, etc.
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)

        ## init empty graph and output buffer? doesnt seem to be used anywhere
        ### PARTIALLY CORRECT: These dicts are initialized empty here but they ARE used
        ### in the regular CudaGraphRunner (for decode). In PiecewiseCudaGraphRunner,
        ### the graph capture/replay is handled differently through torch.compile's
        ### piecewise backend (cuda_piecewise_backend.py), which manages its own graph storage.
        ### These might be legacy or for future use/compatibility.
        self.graphs = {}
        self.output_buffers = {}

        ## getting args from parent
        ### CORRECT
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        set_torch_compile_config()

        ## Needs piecewise_cuda_graph_tokens to be set, altho its auto generated too in server_args.py
        ### CORRECT: piecewise_cuda_graph_tokens defines which token counts to capture graphs for.
        ### Auto-generated in server_args.py based on piecewise_cuda_graph_max_tokens.
        assert (
            self.model_runner.server_args.piecewise_cuda_graph_tokens is not None
        ), "piecewise_cuda_graph_tokens is not set"
        assert self.model_runner.server_args.piecewise_cuda_graph_compiler in [
            "eager",
            "inductor",
        ], "By now, only eager and inductor are supported for piecewise cuda graph compiler."
        self.compile_config = CompilationConfig(
            self.model_runner.server_args.piecewise_cuda_graph_tokens,
            self.model_runner.server_args.piecewise_cuda_graph_compiler,
            self.model_runner.server_args.enable_torch_compile_debug_mode,
        )
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.compile_config.add_split_op(
                "sglang.moe_forward_piecewise_cuda_graph_impl"
            )
            ### MOE all-to-all backends (DeepEP, Mooncake) require special handling.
            ### The split_op tells the piecewise backend where to "cut" the graph,
            ### because these MOE operations can't be captured in CUDA graphs directly.

        self.quant_config = getattr(self.model_runner.model, "quant_config", None)

        # Batch sizes to capture
        self.capture_num_tokens = self.compile_config.get_capture_sizes()
        log_info_on_rank0(
            logger, f"Capture cuda graph num tokens {self.capture_num_tokens}"
        )

        ## Which part of forward (prefill/decode/extend) and hidden states (null, last, all) we capturing
        ### CORRECT: ForwardMode.EXTEND is for processing new tokens (prefill or continuation).
        ### CaptureHiddenMode controls whether to return hidden states:
        ### - NULL: don't capture hidden states (default, most efficient)
        ### - LAST: capture only last token's hidden state
        ### - FULL: capture all tokens' hidden states (needed for some features)
        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        self.max_num_tokens = max(self.capture_num_tokens)

        ## Max buffer size? batch size? seems like batch size
        ### CLARIFICATION: req_to_token_pool.size is the maximum number of REQUESTS
        ### that can be in flight, not batch size directly. It's used here as an upper
        ### bound for batch-dimension allocations. Each request can have multiple tokens,
        ### but this limits how many separate sequences can be processed.
        self.max_bs = model_runner.req_to_token_pool.size

        self.is_multimodal = model_runner.is_multimodal
        self.mamba_track_enabled = self.is_mamba_track_enabled()

        # Graph inputs
        with torch.device(self.device):

            ## dummy input ids with max bits
            ### CORRECTION: "max tokens" not "max bits". This is a pre-allocated buffer
            ### of token IDs. CUDA graphs require fixed memory addresses, so we allocate
            ### the maximum size upfront and copy actual data into slices of this buffer.
            self.input_ids = torch.zeros((self.max_num_tokens,), dtype=torch.int64)
            
            ## output cache location? what this means
            ### EXPLANATION: out_cache_loc maps each token position to its location in the
            ### KV cache. When generating tokens, we need to know WHERE in the KV cache
            ### to write the new key/value vectors. This tensor tells each token position
            ### which slot in the cache to use. The dtype is int64 (or int32 on NPU).
            self.out_cache_loc = torch.zeros(
                (self.max_num_tokens,), dtype=self._cache_loc_dtype()
            )
            self.out_cache_loc_swa = (
                torch.zeros((self.max_num_tokens,), dtype=torch.int64)
                if model_runner.is_hybrid_swa
                else None
            )
            ### out_cache_loc_swa is for Sliding Window Attention (SWA) in hybrid models
            ### that use both full attention and sliding window attention layers.

            ## What these means for mamba?
            ### EXPLANATION: Mamba is a state-space model (SSM) architecture that maintains
            ### recurrent hidden states instead of attention KV caches.
            ### - mamba_track_indices: Maps requests to their Mamba state buffer slots
            ### - mamba_track_mask: Boolean mask indicating which requests have Mamba states
            ### - mamba_track_seqlens: Sequence lengths for Mamba state updates
            ### These enable efficient state management when using radix cache with Mamba.
            self.mamba_track_indices = (
                torch.zeros((self.max_bs,), dtype=torch.int64)
                if self.mamba_track_enabled
                else None
            )
            self.mamba_track_mask = (
                torch.zeros((self.max_bs,), dtype=torch.bool)
                if self.mamba_track_enabled
                else None
            )
            self.mamba_track_seqlens = (
                torch.zeros((self.max_bs,), dtype=torch.int32)
                if self.mamba_track_enabled
                else None
            )
            self.positions = torch.zeros((self.max_num_tokens,), dtype=torch.int64)

            ## Two batch overlap, what this means?
            ### EXPLANATION: TBO (Two Batch Overlap) is an optimization technique that
            ### overlaps the computation of two batches. While one batch is doing attention
            ### (memory-bound), another can do MLP computation (compute-bound).
            ### This maximizes GPU utilization by hiding memory latency with computation.
            ### The plugin prepares the necessary buffers and state for this overlap.
            self.tbo_plugin = TboCudaGraphRunnerPlugin()

            if (
                self.is_multimodal
            ):  # Only create input_embeds and mrope_positions for multimodal model to save memory
                # 1. In multimodal, we only compile and capture the language model part.
                # 2. The embedder is outside of the graph, but cuda graph requires the input embeds to have a fixed memory address.
                # 3. Input embeds is a pre-allocated buffer. In model.forward, we copy the embed output to this buffer.
                self.input_embeds = torch.zeros(
                    (self.max_num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )
                self.mrope_positions = torch.zeros(
                    (3, self.max_num_tokens), dtype=torch.int64
                )
                ### mrope_positions is for Multi-modal Rotary Position Embedding.
                ### The 3 dimensions are typically for (temporal, height, width) positions
                ### in vision-language models like Qwen-VL.
        
        ## Before coming here, we made of list of these type of layers in model runner class
        ### CORRECT: init_piecewise_cuda_graphs() in ModelRunner collects attention_layers
        ### and moe_layers by traversing the model. These are passed to set_forward_context()
        ### so the piecewise backend knows which layers need special handling.
        self.attention_layers = self.model_runner.attention_layers
        self.moe_layers = self.model_runner.moe_layers

        ## cuda graph runner has the same, they use globally set graph memory pool, this in the gpu right?
        ### CORRECT: The graph memory pool IS on the GPU. CUDA graphs need contiguous memory
        ### allocations that persist across replays. graph_pool_handle() returns a handle to
        ### a GPU memory pool that can be shared across multiple graph captures, reducing
        ### fragmentation and total memory usage.
        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())

        ## These are context manager so all process can know what stage we are in right now, when this block is over
        ## The global flag will be set back to false
        ### CORRECT: enable_piecewise_cuda_graph() sets a global flag that other parts of
        ### the code check to know if piecewise CUDA graph mode is active. This affects
        ### how operations are recorded and executed.
        with enable_piecewise_cuda_graph():

            ## We will first make the all the modules in model enter compile stage, then when block finishes
            ## and we go out of context modules will exit compile stage
            ### CORRECT: patch_model puts MultiPlatformOp layers into compile mode where they
            ### may use different (torch.compile-friendly) implementations of operations.
            with patch_model(
                self.model_runner.model.model, self.compile_config.compiler
            ) as patched_model:
                
                ## makes the model compile on first run, then after that will use the compiled version
                ## for future forward passes, this inits sglang compilation backend which has piecewise backend
                ### CORRECT: install_torch_compiled wraps the model with torch.compile using SGLang's
                ### custom backend. The piecewise backend (in cuda_piecewise_backend.py) intercepts
                ### the compilation to:
                ### 1. Split the graph at operations that can't be captured (attention, MOE)
                ### 2. Capture CUDA graphs for the capturable segments
                ### 3. Handle replay by running graphs + dynamic ops in sequence
                install_torch_compiled(
                    patched_model,
                    fullgraph=True,
                    dynamic_arg_dims=None,
                    compile_config=self.compile_config,
                    graph_pool=get_global_graph_memory_pool(),
                )

                with set_compiled(True), enable_piecewise_cuda_graph_compile():
                    compile_range = (
                        tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                        if get_tensor_model_parallel_rank() == 0
                        else reversed(self.capture_num_tokens)
                    )
                    for _, num_tokens in enumerate(compile_range):
                        if get_tensor_model_parallel_rank() == 0:
                            compile_range.set_description(
                                f"Compiling num tokens ({num_tokens=})"
                            )
                        ## why we need a warmup here?
                        ### EXPLANATION: Warmup serves two purposes:
                        ### 1. torch.compile is lazy - it only traces/compiles on first actual execution
                        ### 2. We need to "prime" the compiled code path for each token count BEFORE
                        ###    capturing CUDA graphs, because the compilation happens during warmup.
                        ### Without warmup, the capture phase would trigger compilation mid-capture,
                        ### which can cause issues with CUDA graph recording.
                        self.warmup_torch_compile(num_tokens=num_tokens)
                
                ## Why setting global graph memory pool again?
                ### EXPLANATION: After torch.compile warmup, the device module may have been
                ### reconfigured or the pool handle may have changed. This ensures we have
                ### a fresh, valid pool handle for the actual CUDA graph capture phase.
                ### It's a safety measure to ensure consistency.
                set_global_graph_memory_pool(self.device_module.graph_pool_handle())
                set_graph_pool_id(get_global_graph_memory_pool())

                ## What does the synchronize do?
                ### EXPLANATION: 
                ### - device_module.synchronize(): Waits for all GPU operations to complete.
                ###   Ensures all warmup computations are done before proceeding.
                ### - tp_group.barrier(): Synchronizes across tensor parallel ranks. In multi-GPU
                ###   setups, all GPUs must reach this point before any can proceed. This prevents
                ###   race conditions during graph capture where GPUs might be at different stages.
                self.device_module.synchronize()
                self.model_runner.tp_group.barrier()
                # Capture
                try:
                    self.capture()
                except RuntimeError as e:
                    raise Exception(
                        f"Capture cuda graph failed: {e}\n{PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG}"
                    )

        self.raw_num_tokens = 0

    def warmup_torch_compile(self, num_tokens: int):
        """Warmup the model with a simple forward pass before CUDA graph capture."""
        ### This creates a minimal ForwardBatch with dummy data to trigger torch.compile
        ### tracing for the given num_tokens. After this, the compiled code is cached.
        input_ids = self.input_ids[:num_tokens]
        input_embeds = self.input_embeds[:num_tokens] if self.is_multimodal else None
        positions = self.positions[:num_tokens]
        mrope_positions = (
            self.mrope_positions[:, :num_tokens] if self.is_multimodal else None
        )
        out_cache_loc = self.out_cache_loc[:num_tokens]
        out_cache_loc_swa = (
            self.out_cache_loc_swa[:num_tokens]
            if self.out_cache_loc_swa is not None
            else None
        )
        mamba_track_indices = (
            self.mamba_track_indices[:1]
            if self.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            self.mamba_track_mask[:1] if self.mamba_track_mask is not None else None
        )
        mamba_track_seqlens = (
            self.mamba_track_seqlens[:1]
            if self.mamba_track_seqlens is not None
            else None
        )
        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=input_ids,
                input_embeds=input_embeds,
                req_pool_indices=torch.arange(1, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                out_cache_loc_swa=out_cache_loc_swa,
                seq_lens_sum=num_tokens,
                mamba_track_indices=mamba_track_indices,
                mamba_track_mask=mamba_track_mask,
                mamba_track_seqlens=mamba_track_seqlens,
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([num_tokens], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                positions=positions,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=mrope_positions,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
            )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(None, num_tokens, forward_batch.dp_padding_mode.is_max_len())
        set_is_extend_in_batch(False)
        with set_forward_context(
            forward_batch, self.attention_layers, self.quant_config, self.moe_layers
        ):
            _ = self.model_runner.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

    def _cache_loc_dtype(self):
        return torch.int64 if not is_npu() else torch.int32
        ### NPU (Huawei Ascend) uses int32 for cache locations, likely due to
        ### hardware constraints or optimization reasons.

    def can_run(self, forward_batch: ForwardBatch):
        ### Determines if this batch can use piecewise CUDA graphs
        num_tokens = len(forward_batch.input_ids)
        if forward_batch.return_logprob:
            for start_len, seq_len in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
            ):
                if start_len is not None and start_len < seq_len:
                    return False
                    ### If logprobs are needed for tokens in the middle of a sequence
                    ### (not just the last token), we can't use CUDA graphs because
                    ### the logprob computation path is different and not captured.
        if num_tokens <= self.max_num_tokens:
            return True
        return False

    def capture(self) -> None:
        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.

        ## freezing the gc collection so it doesnt interfere with graph capturng
        ### CORRECT
        with freeze_gc(
            self.model_runner.server_args.enable_cudagraph_gc
        ), graph_capture() as graph_capture_context:
            stream = graph_capture_context.stream
            ### graph_capture() sets up a dedicated CUDA stream for capture.
            ### CUDA graph capture records operations on a specific stream.
            with set_pcg_capture_stream(stream):
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                # Reverse the order to enable better memory sharing across cuda graphs.
                ### Capturing larger graphs first allows smaller graphs to reuse
                ### memory allocated for larger ones (subgraph memory sharing).
                capture_range = (
                    tqdm.tqdm(list(reversed(self.capture_num_tokens)))
                    if get_tensor_model_parallel_rank() == 0
                    else reversed(self.capture_num_tokens)
                )
                for i, num_tokens in enumerate(capture_range):
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_range.set_description(
                            f"Capturing num tokens ({num_tokens=} {avail_mem=:.2f} GB)"
                        )

                    with set_compiled(True):
                        self.capture_one_batch_size(num_tokens)

    def capture_one_batch_size(self, num_tokens: int):
        bs = 1
        ### Note: Piecewise CUDA graph captures with bs=1 but variable num_tokens.
        ### This is different from decode CUDA graphs which vary batch size.
        ### For extend/prefill, the token count is the more important dimension.

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        input_embeds = self.input_embeds[:num_tokens] if self.is_multimodal else None

        out_cache_loc = self.out_cache_loc[:num_tokens]
        out_cache_loc_swa = (
            self.out_cache_loc_swa[:num_tokens]
            if self.out_cache_loc_swa is not None
            else None
        )
        mamba_track_indices = (
            self.mamba_track_indices[:bs]
            if self.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            self.mamba_track_mask[:bs] if self.mamba_track_mask is not None else None
        )
        mamba_track_seqlens = (
            self.mamba_track_seqlens[:bs]
            if self.mamba_track_seqlens is not None
            else None
        )
        positions = self.positions[:num_tokens]
        mrope_positions = (
            self.mrope_positions[:, :num_tokens] if self.is_multimodal else None
        )

        global_dp_buffer_len = None

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        with torch.device(self.device):
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=bs,
                input_ids=input_ids,
                input_embeds=input_embeds,
                req_pool_indices=torch.arange(bs, device=self.device),
                seq_lens=torch.tensor([num_tokens], device=self.device),
                next_token_logits_buffer=None,
                orig_seq_lens=torch.tensor([num_tokens], device=self.device),
                seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool=self.model_runner.token_to_kv_pool,
                attn_backend=self.model_runner.attn_backend,
                out_cache_loc=out_cache_loc,
                out_cache_loc_swa=out_cache_loc_swa,
                seq_lens_sum=num_tokens,
                mamba_track_indices=mamba_track_indices,
                mamba_track_mask=mamba_track_mask,
                mamba_track_seqlens=mamba_track_seqlens,
                encoder_lens=None,
                return_logprob=False,
                extend_num_tokens=num_tokens,
                extend_seq_lens=torch.tensor([num_tokens], device=self.device),
                extend_prefix_lens=torch.tensor([num_tokens], device=self.device),
                extend_start_loc=torch.tensor([0], device=self.device),
                extend_prefix_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_seq_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                extend_logprob_start_lens_cpu=torch.tensor([num_tokens], device="cpu"),
                positions=positions,
                global_num_tokens_gpu=None,
                global_num_tokens_for_logprob_gpu=None,
                dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
                global_dp_buffer_len=None,
                mrope_positions=mrope_positions,
                spec_algorithm=None,
                spec_info=None,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                num_token_non_padded=None,
                global_forward_mode=ForwardMode.EXTEND,
                lora_ids=None,
            )
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            # FIXME: the implementation is hacky. `is_extend_in_batch`` is for determining the deepep mode.
            # It is True in this context but we need to set it to use low latency deepep mode.
            set_is_extend_in_batch(False)

            kwargs = {}
            with set_forward_context(
                forward_batch, self.attention_layers, self.quant_config, self.moe_layers
            ):
                self.model_runner.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
            return

        # run twice for warmup at the first time and cuda graph capture at the second time
        # detail lies in sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py
        ### The piecewise backend uses a state machine:
        ### - 1st run: Records which ops to capture vs run dynamically
        ### - 2nd run: Actually captures the CUDA graphs for capturable segments
        ### This two-phase approach is necessary because the backend needs to
        ### analyze the computation graph before knowing how to partition it.
        for _ in range(2):
            ## what is this sync + barrier?
            ### EXPLANATION:
            ### - synchronize(): Ensures all pending GPU work completes before next iteration.
            ###   Critical between warmup and capture phases to have clean state.
            ### - barrier(): Ensures all tensor parallel ranks are synchronized.
            ###   All GPUs must complete warmup before any starts capture, otherwise
            ###   collective operations (like all-reduce) would hang.
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        return

    # ... [replay_prepare and replay methods continue similarly]
```

python/sglang/srt/compilation/compile.py

```python
import contextvars
import inspect
import logging
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.utils.common import rank0_log

logger = logging.getLogger(__name__)

### OVERVIEW:
### This file provides the core infrastructure for integrating torch.compile with SGLang.
### torch.compile is PyTorch's JIT compiler that can optimize models by:
### 1. Tracing the computation graph (via TorchDynamo)
### 2. Lowering to optimized kernels (via backends like Inductor)
### 
### SGLang uses a custom backend (SGLangBackend) that enables "piecewise" compilation,
### where the model is split at dynamic operations (attention, MOE) and static parts
### are captured as CUDA graphs.

_COMPILE_ENABLED = contextvars.ContextVar("_COMPILE_ENABLED", default=False)
### contextvars.ContextVar is used instead of a simple global variable because:
### 1. It's thread-safe and async-safe (each async task/thread gets its own copy)
### 2. It supports proper scoping with tokens for nested contexts
### 3. Essential when multiple requests might be processed concurrently

## setter for global variable _COMPILE_ENABLED
### CORRECT: This is a context manager that temporarily enables/disables compiled mode.
### When inside `with set_compiled(True):`, the compiled forward path is used.
### The token mechanism ensures proper nesting (can have set_compiled inside set_compiled).
@contextmanager
def set_compiled(enabled: bool = True):
    token = _COMPILE_ENABLED.set(enabled)
    try:
        yield
    finally:
        _COMPILE_ENABLED.reset(token)


@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.

    Each stage also needs to handle its own finished_sending and
    finished_recving in case of kv transfer.
    """
    ### This is used in Pipeline Parallelism (PP) where different layers run on different GPUs.
    ### Between stages, we need to pass:
    ### - "hidden_states": the main activation tensor
    ### - "residual": for residual connections in transformers
    ### The finished_sending/recving track KV cache transfer status for disaggregated prefill.

    tensors: dict[str, torch.Tensor]
    # [req_ids]
    finished_sending: Optional[set[str]] = None
    finished_recving: Optional[set[str]] = None

    def __init__(self, tensors):
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        ### IMPORTANT: TorchDynamo traces Python bytecode and needs to know where
        ### code comes from for proper graph breaks and caching. Dataclass-generated
        ### __init__ uses exec() which loses source file info, causing Dynamo issues.
        self.tensors = tensors

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})
        ### Slice support allows: intermediate_tensors[start:end] to slice ALL contained
        ### tensors along their batch dimension simultaneously.

    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    def items(self):
        return self.tensors.items()

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self
        ### Note: This returns `self` (truthy if tensors exist) when types match.
        ### Somewhat unusual - doesn't compare tensor contents, just checks type.

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"


def _normalize_dims(dims, ndim: int):
    ### Converts negative dimension indices to positive ones.
    ### e.g., for a 3D tensor: dim=-1 becomes dim=2
    ### This is needed because torch._dynamo.mark_dynamic expects positive indices.
    dims = [dims] if isinstance(dims, int) else list(dims)
    return [d if d >= 0 else ndim + d for d in dims]


class _MaybeIntermediateTensors:
    """Duck-typed check to support your IntermediateTensors without importing."""
    ### Duck typing avoids circular imports and allows this to work with any object
    ### that has a `tensors` dict attribute, not just IntermediateTensors specifically.

    def __init__(self, obj):
        self.is_intermediate = hasattr(obj, "tensors") and isinstance(
            getattr(obj, "tensors"), dict
        )
        self.obj = obj


def _mark_dynamic_on_value(val, dims):
    ### torch._dynamo.mark_dynamic tells TorchDynamo that certain tensor dimensions
    ### can vary at runtime. Without this, Dynamo assumes shapes are static and
    ### would recompile for every new shape (very slow).
    ### 
    ### For LLMs, the batch dimension (dim 0) is typically dynamic because:
    ### - Different numbers of requests in a batch
    ### - Different sequence lengths during prefill
    if isinstance(val, torch.Tensor):
        torch._dynamo.mark_dynamic(val, _normalize_dims(dims, val.ndim))
    else:
        mit = _MaybeIntermediateTensors(val)
        if mit.is_intermediate:
            for t in mit.obj.tensors.values():
                torch._dynamo.mark_dynamic(t, _normalize_dims(dims, t.ndim))
        # else: ignore (None or non-tensor)


## what kind of args is this getting?
### EXPLANATION: This function inspects the type annotations of a forward() method
### to automatically determine which arguments have dynamic dimensions.
### 
### For example, given:
###   def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ...):
### 
### It will return: {"input_ids": 0, "positions": 0}
### meaning dimension 0 (batch/sequence dim) is dynamic for these tensor args.
### 
### This is a convenience feature so users don't have to manually specify
### dynamic_arg_dims for every tensor argument.
def _infer_dynamic_arg_dims_from_annotations(forward_fn):
    sig = inspect.signature(forward_fn)
    dyn = {}
    for name, p in sig.parameters.items():
        ann = p.annotation
        # Accept torch.Tensor / Optional[torch.Tensor] / your IntermediateTensors types by name
        if (
            ann is torch.Tensor
            or getattr(getattr(ann, "__args__", [None])[0], "__name__", "") == "Tensor"
            ### The second condition handles Optional[torch.Tensor] where __args__[0] is Tensor
        ):
            dyn[name] = 0
        elif getattr(ann, "__name__", "") in ("IntermediateTensors",) or any(
            getattr(a, "__name__", "") == "IntermediateTensors"
            for a in getattr(ann, "__args__", [])
        ):
            dyn[name] = 0
        elif ann == "torch.Tensor" or ann == "Optional[torch.Tensor]":
            # For future import annotations (e.g. from __future__ import annotations), the annotation is a string
            ### With `from __future__ import annotations`, annotations become strings
            ### instead of actual type objects, so we need string comparison too.
            dyn[name] = 0
    if not dyn:
        raise ValueError("No dynamic dims inferred; pass dynamic_arg_dims explicitly.")
    return dyn


## JIT Compilation for torch modules with SGLang backend
### CORRECT: This is the main entry point for enabling torch.compile on a model.
### It wraps the module's forward method with a "trampoline" that can switch
### between compiled and uncompiled execution based on the _COMPILE_ENABLED flag.
def install_torch_compiled(
    module: torch.nn.Module,
    *,
    dynamic_arg_dims: dict[str, Union[int, list[int]]] | None = None,
    ### dynamic_arg_dims: Maps argument names to which dimensions are dynamic
    ### e.g., {"input_ids": 0, "hidden_states": [0, 1]} means input_ids has
    ### dynamic dim 0, hidden_states has dynamic dims 0 and 1.
    backend_factory: Optional[Callable[[torch.fx.GraphModule, list], Callable]] = None,
    ### backend_factory: Custom compilation backend. If None, uses SGLangBackend.
    ### The factory receives the traced FX graph and returns a callable.
    compile_config: CompilationConfig = None,
    fullgraph: bool = True,
    ### fullgraph=True means Dynamo should capture the ENTIRE forward pass as one graph.
    ### If it can't (due to unsupported ops), it raises an error instead of silently
    ### inserting graph breaks. This is important for piecewise CUDA graphs.
    graph_pool: Any = None,
    ### graph_pool: Shared CUDA graph memory pool for memory efficiency.
):
    rank0_log(f"install_torch_compiled")

    ## Non compiled fwd method
    ### CORRECT: We get the unbound method from the class (not instance) so we can
    ### properly wrap it while keeping access to both the original and compiled versions.
    unbound_fwd = module.__class__.forward

    if not callable(unbound_fwd):
        raise TypeError("module.__class__.forward must be callable")
    original_code = unbound_fwd.__code__
    ### __code__ is the Python code object containing the actual bytecode.
    ### We save this to identify when Dynamo compiles THIS specific function.

    dyn_map = dynamic_arg_dims or _infer_dynamic_arg_dims_from_annotations(unbound_fwd)

    ## What kind of compilation we are doing, based on backend, piecewise or full? other types maybe there
    ### EXPLANATION: The backend determines HOW the traced graph gets compiled:
    ### - SGLangBackend: Implements piecewise CUDA graph capture. It analyzes the graph,
    ###   identifies "split points" (attention, MOE), and captures CUDA graphs for
    ###   the static segments between split points.
    ### - "inductor": PyTorch's default backend, generates Triton kernels
    ### - "eager": No compilation, just traces and runs (useful for debugging)
    ### 
    ### SGLangBackend is in sglang/srt/compilation/backend.py and delegates to
    ### cuda_piecewise_backend.py for the actual piecewise capture logic.
    if backend_factory is None:
        from sglang.srt.compilation.backend import SGLangBackend

        backend_factory = lambda gm, ex: SGLangBackend(compile_config, graph_pool)(
            gm, ex
        )

    compiled_codes: list[type(original_code)] = []
    state = {"compiled": False, "compiled_callable": None}
    ### Using a dict for state instead of simple variables because:
    ### 1. Closures in Python capture variables by reference for mutable containers
    ### 2. Simple variables would be captured by value, making updates invisible

    ## Find out what this hook is doing?
    ### EXPLANATION: This bytecode hook intercepts TorchDynamo's compilation process.
    ### 
    ### When torch.compile runs, Dynamo:
    ### 1. Traces the Python function's bytecode
    ### 2. Builds an FX graph representation
    ### 3. Compiles it with the backend
    ### 4. Generates NEW bytecode that calls the compiled version
    ### 
    ### This hook is called when Dynamo generates that new bytecode (new_code).
    ### The hook:
    ### 1. Checks if the old_code being replaced is our original forward method
    ### 2. Walks up the call stack to find Dynamo's internal frame
    ### 3. Verifies this compilation is for OUR specific module instance
    ### 4. Saves the compiled bytecode for later reference
    ### 
    ### This allows SGLang to track which compilations happened and potentially
    ### manage multiple compiled versions for different configurations.
    def bytecode_hook(old_code, new_code):
        if old_code is not original_code:
            return
            ### Ignore compilations of other functions
        frame = sys._getframe()
        while frame and frame.f_back:
            frame = frame.f_back
            if (
                frame.f_code.co_name == "_compile"
                and os.path.basename(frame.f_code.co_filename) == "convert_frame.py"
            ):
                break
                ### Found Dynamo's _compile function in convert_frame.py
        try:
            dynamo_frame = frame.f_locals["frame"]
            ### Get the frame being compiled from Dynamo's local variables
        except Exception:
            return
        if dynamo_frame.f_code is not old_code:
            return
        if dynamo_frame.f_locals.get("self") is not module:
            return
            ### Ensure this is compiling OUR module, not some other instance
        compiled_codes.append(new_code)
    
    ## Where this hook latches on to?
    ### EXPLANATION: This registers the hook with TorchDynamo's bytecode transformation
    ### system. Dynamo uses Python's frame evaluation hooks (PEP 523) to intercept
    ### function calls. When it decides to compile a function:
    ### 1. It traces the bytecode to build an FX graph
    ### 2. Compiles the graph with the backend
    ### 3. Generates new bytecode that uses the compiled version
    ### 4. Calls all registered bytecode_hooks with (old_code, new_code)
    ### 
    ### The hook is global - it will be called for ALL Dynamo compilations,
    ### which is why we filter to only care about our specific module.
    torch._dynamo.convert_frame.register_bytecode_hook(bytecode_hook)

    def _ensure_compiled(self, *args, **kwargs):
        """Compile on first use (with flag ON)."""
        ### This implements LAZY compilation - we don't compile until the model
        ### is actually used. This is important because:
        ### 1. We need real tensor shapes to properly mark dynamic dimensions
        ### 2. Some model configurations might not be known until runtime
        ### 3. Avoids compiling code paths that are never used
        if state["compiled"]:
            return
        # Mark dynamic dims only when we are about to compile
        sig = inspect.signature(unbound_fwd)
        ba = sig.bind(self, *args, **kwargs)
        ba.apply_defaults()
        ### bind() matches args/kwargs to parameter names
        ### apply_defaults() fills in default values for unspecified params
        for name, dims in (dyn_map or {}).items():
            if name in ba.arguments:
                val = ba.arguments[name]
                if val is not None:
                    _mark_dynamic_on_value(val, dims)
                    ### Mark tensors as having dynamic dimensions BEFORE tracing

        # Avoid cross-instance cache reuse
        ### Dynamo caches compiled code based on the code object. If we have
        ### multiple model instances (e.g., draft and target models in speculative
        ### decoding), we don't want them sharing cached compilations since they
        ### might have different configurations. This clears any existing cache.
        torch._dynamo.eval_frame.remove_from_cache(unbound_fwd.__code__)

        bound = types.MethodType(unbound_fwd, self)
        ### Create a bound method (method + instance) for torch.compile
        compiled_callable = torch.compile(
            bound, fullgraph=fullgraph, backend=backend_factory
        )

        # Trigger Dynamo so bytecode hook can capture
        ### The first call to compiled_callable actually triggers compilation.
        ### torch.compile is lazy - it doesn't compile until first execution.
        ### This call traces the forward pass and invokes our bytecode_hook.
        compiled_callable(*args, **kwargs)

        state["compiled"] = True
        state["compiled_callable"] = compiled_callable
    
    ## This is where it checks if compiled, the runs the compiled version else compiles it first
    ## If we dont want compiled then return the original code (unbound forward)
    ### CORRECT: The "trampoline" pattern provides a single entry point that can
    ### dispatch to either compiled or uncompiled execution.
    ### 
    ### Why "trampoline"? In programming, a trampoline is a function that bounces
    ### control to different implementations based on runtime conditions.
    ### 
    ### Benefits:
    ### 1. Seamless switching between compiled/uncompiled (for debugging, profiling)
    ### 2. Lazy compilation (compile only when _COMPILE_ENABLED is True)
    ### 3. Same API regardless of compilation state
    def trampoline(self, *args, **kwargs):
        use_compiled = _COMPILE_ENABLED.get()
        if use_compiled:
            if not state["compiled"]:
                _ensure_compiled(self, *args, **kwargs)

            compiled_callable = state["compiled_callable"]
            return compiled_callable(*args, **kwargs)
        else:
            # Explicitly run the original uncompiled forward
            return unbound_fwd(self, *args, **kwargs)

    module.forward = types.MethodType(trampoline, module)
    ### Replace the module's forward method with our trampoline.
    ### types.MethodType binds the trampoline function to the module instance,
    ### so `self` in trampoline refers to the module.
    return module
```


```
User calls model.forward()
        │
        ▼
    trampoline()
        │
        ├─── _COMPILE_ENABLED = False ──► Original forward()
        │
        └─── _COMPILE_ENABLED = True
                    │
                    ├─── Not compiled yet ──► _ensure_compiled()
                    │                               │
                    │                               ▼
                    │                         torch.compile()
                    │                               │
                    │                               ▼
                    │                         SGLangBackend
                    │                               │
                    │                               ▼
                    │                    Piecewise CUDA Graph Capture
                    │
                    └─── Already compiled ──► compiled_callable()
```



```python
# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/backend.py

### OVERVIEW:
### This file implements SGLang's custom torch.compile backend. When torch.compile is called,
### PyTorch's TorchDynamo traces the model and produces an FX graph. This backend then:
### 1. Splits the graph at "split ops" (attention, MOE) that can't be captured in CUDA graphs
### 2. Compiles the capturable segments using Inductor or Eager mode
### 3. Wraps each segment with CUDAPiecewiseBackend for CUDA graph capture
### 
### The result is a hybrid execution where:
### - Static segments run as CUDA graphs (fast, low overhead)
### - Dynamic segments (attention) run normally with FlashAttention etc.

import ast
import dataclasses
import logging
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compiler_interface import EagerAdapter, InductorAdaptor
from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend
from sglang.srt.compilation.npu_piecewise_backend import NPUPiecewiseBackend
from sglang.srt.compilation.pass_manager import PostGradPassManager
from sglang.srt.utils.common import is_npu, rank0_log

logger = logging.getLogger(__name__)


def make_compiler(config: CompilationConfig):
    ### Creates the actual compiler that will lower FX graphs to executable code.
    ### - EagerAdapter: No optimization, just wraps the graph for direct execution (debugging)
    ### - InductorAdaptor: Uses PyTorch Inductor to generate optimized Triton/CUDA kernels
    if config.compiler == "eager":
        return EagerAdapter()
    elif config.compiler == "inductor":
        return InductorAdaptor()
    else:
        raise ValueError(f"Unknown compiler: {config.compiler}")


## Choose backend for cuda/npu for piecewise respectively
### CORRECT: This factory function creates the appropriate piecewise backend.
### CUDAPiecewiseBackend handles CUDA graph capture for NVIDIA GPUs.
### NPUPiecewiseBackend handles equivalent functionality for Huawei Ascend NPUs.
def make_backend(
    graph: fx.GraphModule,
    compile_config: CompilationConfig,
    inductor_config: dict[str, Any],
    graph_pool: Any,
    piecewise_compile_index: int,
    ### piecewise_compile_index: Which piece this is (0, 1, 2, ...)
    total_piecewise_compiles: int,
    ### total_piecewise_compiles: Total number of capturable segments
    sym_shape_indices: list[int],
    ### sym_shape_indices: Indices of arguments that are symbolic (dynamic) shapes
    compiled_graph_for_general_shape: Callable,
    ### Fallback compiled graph for shapes not captured as CUDA graphs
    sglang_backend,
):

    backend_cls = CUDAPiecewiseBackend if not is_npu() else NPUPiecewiseBackend
    return backend_cls(
        graph,
        compile_config,
        inductor_config,
        graph_pool,
        piecewise_compile_index,
        total_piecewise_compiles,
        sym_shape_indices,
        compiled_graph_for_general_shape,
        sglang_backend,
    )


## This compiles, saves and loads compiled code
### CORRECT: CompilerManager handles the compilation lifecycle:
### 1. Caching - saves compiled artifacts to disk for faster startup
### 2. Loading - retrieves previously compiled code
### 3. Compiling - invokes the actual compiler when cache misses
class CompilerManager:
    def __init__(
        self,
        config: CompilationConfig,
    ):
        self.cache = dict()
        ### Cache maps (runtime_shape, graph_index, compiler_name) -> handle
        ### The handle is used to retrieve the compiled artifact from disk
        self.is_cache_updated = False
        self.compiler = make_compiler(config)

    def compute_hash(self):
        ### Hash for cache invalidation - if model/config changes, hash changes
        return self.compiler.compute_hash()

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "sglang_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())
                ### ast.literal_eval safely parses Python literals from file
                ### The cache file is a Python dict written as a string

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Optional[Callable]:
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via " "handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        inductor_config: dict[str, Any],
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
        ### runtime_shape: If None, compiles for dynamic/symbolic shapes
        ### If an int, compiles a specialized version for that exact shape
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # TODO(Yuwei): support cache loading

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            maybe_key = None
            ### Inductor has its own caching mechanism, so we don't need a key
        else:
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, inductor_config, runtime_shape, maybe_key
        )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if handle is not None:
            self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                if runtime_shape is None:
                    logger.info("Cache the graph for dynamic shape for later use")
                else:
                    logger.info(
                        "Cache the graph of shape %s for later use", str(runtime_shape)
                    )
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via " "handle %s",
                    graph_index,
                    self.compiler.name,
                    handle,
                )
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index,
                    str(runtime_shape),
                    self.compiler.name,
                    handle,
                )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            if runtime_shape is None:
                logger.info("Compiling a graph for dynamic shape takes %.2f s", elapsed)
            else:
                logger.info(
                    "Compiling a graph for shape %s takes %.2f s",
                    runtime_shape,
                    elapsed,
                )

        return compiled_graph


@dataclasses.dataclass
class SplitItem:
    """Represents one piece of the split computation graph."""
    submod_name: str          # e.g., "submod_0", "submod_1"
    graph_id: int             # Integer ID for ordering
    is_splitting_graph: bool  # True if this is a split point (attention/MOE)
    graph: fx.GraphModule     # The actual subgraph


def split_graph(
    graph: fx.GraphModule, ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    ### This function splits an FX graph at specified operations.
    ### 
    ### For example, if the graph is:
    ###   embed -> layernorm -> attention -> mlp -> layernorm -> attention -> mlp -> logits
    ### 
    ### And ops = ["attention"], it splits into:
    ###   submod_0: embed -> layernorm
    ###   submod_1: attention (split point, NOT compiled into CUDA graph)
    ###   submod_2: mlp -> layernorm
    ###   submod_3: attention (split point)
    ###   submod_4: mlp -> logits
    ###
    ### The even-numbered submods (0, 2, 4) are capturable as CUDA graphs.
    ### The odd-numbered submods (1, 3) are split points that run dynamically.
    
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
            ### Skip input (placeholder) and output nodes - they're structural
        if node.op == "call_function" and str(node.target) in ops:
            ### Found a split operation (e.g., attention)
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            ### The split op gets its own subgraph ID
            subgraph_id += 1
            ### Increment again so next ops go to a new subgraph
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    ### CRITICAL: Without keep_original_order, PyTorch might reorder operations
    ### for "optimization", but if there are in-place mutations (like updating
    ### KV cache), reordering would break correctness.
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    ### Important because "submod_10" < "submod_2" alphabetically but 2 < 10 numerically
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


## What does interpreter means here? 
### EXPLANATION: torch.fx.Interpreter is a class that executes FX graphs node-by-node.
### Instead of running the whole graph at once, it:
### 1. Iterates through each node in topological order
### 2. Calls the appropriate method (call_function, call_module, etc.) for each node
### 3. Tracks intermediate values
###
### By subclassing Interpreter, we can INTERCEPT specific operations.
### Here, we intercept call_module to:
### - Identify submodules that should be compiled
### - Compile them and replace them with CUDAPiecewiseBackend wrappers
###
### Think of it as a "visitor pattern" for FX graphs - we visit each node
### and do custom processing for the ones we care about.

## Args: Graph module, list of compiled sub modules, inductor config (idk what inductor is)
### CORRECTION on "inductor config (idk what inductor is)":
### 
### INDUCTOR is PyTorch's default torch.compile backend. It:
### 1. Takes an FX graph (high-level ops like matmul, softmax)
### 2. Lowers it to Triton IR (GPU kernel language)
### 3. Generates optimized CUDA kernels
###
### inductor_config is a dict of settings like:
### - "enable_auto_functionalized_v2": Controls handling of in-place ops
### - "max_autotune": Whether to benchmark multiple kernel implementations
### - "coordinate_descent_tuning": Kernel tuning strategy
###
### Inductor is powerful because it can fuse operations (e.g., LayerNorm + Linear)
### into single efficient kernels.

## backend name, graph pool memory
### CORRECT: graph_pool is the shared CUDA graph memory pool on GPU
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        ### Names of submodules to compile (the capturable ones, NOT split points)
        inductor_config: dict[str, Any],
        graph_pool,
        compile_config: CompilationConfig,
        sglang_backend: "SGLangBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        ### Fake mode is PyTorch's mechanism for shape inference without real data.
        ### "Fake tensors" have shapes and dtypes but no actual memory/values.
        ### This allows tracing/compilation without needing real GPU tensors.
        self.compile_submod_names = compile_submod_names
        self.graph_pool = graph_pool
        self.sglang_backend = sglang_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False
        self.inductor_config = inductor_config
        self.compile_config = compile_config

    def run(self, *args):

        ## What are these fake args?
        ### EXPLANATION: Fake tensors are "meta tensors" - they have:
        ### - Shape information (e.g., [batch_size, seq_len, hidden_dim])
        ### - Dtype (e.g., torch.float16)
        ### - Device info (e.g., cuda:0)
        ### But NO actual data (no GPU memory allocated).
        ###
        ### Why use them?
        ### 1. Compilation needs shape info but doesn't need actual values
        ### 2. We can trace through the model without GPU memory overhead
        ### 3. Enables shape inference and validation without computation
        ###
        ### from_tensor() converts real tensors to fake equivalents.
        ### enable_python_dispatcher() enables dispatch to work with fake tensors.
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)
    
    ## This seems like we are compiling and making backend of only target module within the submodule list
    ## But need to deep dive into the syntax
    ### CORRECT! This is the heart of piecewise compilation.
    ###
    ### When the interpreter encounters a call_module node (calling a submodule),
    ### this method is invoked. The logic:
    ### 1. Let PyTorch execute the submodule normally (super().call_module)
    ### 2. If this submodule is in our "to compile" list:
    ###    a. Compile it with Inductor for dynamic shapes (fallback)
    ###    b. Wrap it with CUDAPiecewiseBackend for CUDA graph capture
    ###    c. Replace the original submodule with the wrapped version
    ###
    ### After this runs, split_gm's submodules are replaced with backends
    ### that can capture/replay CUDA graphs!
    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)
        ### First, execute normally to get output shapes for downstream nodes

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            ### fetch_attr gets the actual submodule from the parent graph module
            
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            ### SymInt = Symbolic Integer. These are dynamic dimensions whose
            ### actual values aren't known until runtime (e.g., batch_size, seq_len).
            ### We track which argument positions are symbolic.
            
            global compilation_start_time
            compiled_graph_for_dynamic_shape = (
                self.sglang_backend.compiler_manager.compile(
                    submod,
                    args,
                    self.inductor_config,
                    graph_index=index,
                    num_graphs=len(self.compile_submod_names),
                    runtime_shape=None,  # None = compile for dynamic shapes
                )
            )
            ### This compiles a GENERAL version that works for any shape.
            ### Used as fallback when input shape doesn't match any captured CUDA graph.

            self.module.__dict__[target] = make_backend(
                submod,
                self.compile_config,
                self.inductor_config,
                self.graph_pool,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.sglang_backend,
            )
            ### REPLACE the submodule with CUDAPiecewiseBackend!
            ### Now when split_gm runs, it calls the backend instead of the original submod.
            ### The backend handles CUDA graph capture/replay.

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


model_tag: str = "backbone"
### Tag for cache organization. Different model components (backbone, draft model)
### can have separate cache directories.


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    ### Used when compiling multiple models (e.g., target + draft in speculative decoding)
    ### to keep their caches separate.
    global model_tag
    assert (
        tag != model_tag
    ), f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


## Entry point from the install torch compiled
### CORRECT: This is the main backend class that torch.compile calls.
### When you do torch.compile(model, backend=SGLangBackend(...)), PyTorch:
### 1. Traces the model with TorchDynamo to get an FX graph
### 2. Calls SGLangBackend.__call__(graph, example_inputs)
### 3. Expects a callable back that executes the optimized computation
class SGLangBackend:

    graph_pool: Any
    _called: bool = False
    ### Safety check - each backend instance should only compile once
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    ### split_gm is the "stitched" graph that calls all the submodules in order.
    ### It looks like:
    ###   def forward(x):
    ###       x = self.submod_0(x)  # -> CUDAPiecewiseBackend
    ###       x = self.submod_1(x)  # -> original attention (split point)
    ###       x = self.submod_2(x)  # -> CUDAPiecewiseBackend
    ###       ...
    ###       return x
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable

    ## What inductor means? What is this?
    ### Already explained above, but to summarize:
    ### Inductor = PyTorch's compiler that generates optimized GPU kernels.
    ### These passes are graph transformations that run after autograd captures gradients.
    
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        config: CompilationConfig,
        graph_pool: Any,
    ):
        rank0_log(f"Initializing SGLangBackend")
        assert graph_pool is not None
        self.graph_pool = graph_pool

        ## According to docstring: The pass manager for post-grad passes. It handles configuration, adding custom passes, 
        ## and running passes. It supports uuid for the Inductor code cache
        ## What this means tho?
        ### EXPLANATION: Post-grad passes are optimization transformations that run
        ### AFTER TorchDynamo has traced the graph (which includes autograd information).
        ###
        ### "Post-grad" means after gradient computation is traced (even if we're not
        ### doing training). The passes can:
        ### - Fuse operations (combine multiple ops into one kernel)
        ### - Remove redundant operations
        ### - Optimize memory access patterns
        ### - Apply custom SGLang-specific optimizations
        ###
        ### The "uuid for Inductor code cache" means each unique configuration gets
        ### a unique ID, so compiled kernels can be cached and reused across runs.
        self.post_grad_pass_manager = PostGradPassManager()
        self.sym_tensor_indices = []
        self.input_buffers = []

        ## handles, compiling start, saving and loading of compiled
        ### CORRECT
        self.compiler_manager = CompilerManager(config)
        self.inductor_config = {
            "enable_auto_functionalized_v2": False,
            ### Disables a specific autograd functionalization feature.
            ### "Functionalization" converts in-place ops to out-of-place ops
            ### for easier optimization, but v2 can cause issues with some patterns.
        }
        self.compile_config = config

    ## What is post grad? Gradient? Why it needs a manager
    ### CLARIFICATION: "Post-grad" doesn't mean "after gradient computation" in the
    ### training sense. It means "after the graph has been captured by Dynamo's
    ### grad-aware tracing."
    ###
    ### Even for inference, PyTorch traces the model in a way that's compatible
    ### with autograd (in case someone wants gradients). These passes run on that
    ### traced graph before final compilation.
    ###
    ### Why a manager? Because:
    ### 1. Multiple passes need to run in specific order
    ### 2. Passes need configuration (enable/disable certain optimizations)
    ### 3. Passes need to be composable (SGLang can add custom passes)
    ### 4. Cache management needs consistent UUIDs across passes
    def configure_post_pass(self):
        self.post_grad_pass_manager.configure()
        self.inductor_config["post_grad_custom_post_pass"] = self.post_grad_pass_manager
        ### This injects SGLang's pass manager into Inductor's pipeline

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        ### This is called by torch.compile with the traced FX graph.
        ### Returns the optimized callable (split_gm with backends installed).
        rank0_log(f"SGLangBackend __call__")
        base_cache_dir = os.path.expanduser(
            os.getenv("SGLANG_CACHE_DIR", "~/.cache/sglang/")
        )

        cache_hash = self.compiler_manager.compute_hash()
        cache_dir = os.path.join(
            base_cache_dir,
            "torch_compile_cache",
            cache_hash,
        )

        os.makedirs(cache_dir, exist_ok=True)
        rank = 0
        dp_rank = 0
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", model_tag)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache=False, prefix=""
        )
        compilation_counter.num_graphs_seen += 1

        ## Is this on first run of the model? After that it will only pickup from the cached compiled state?
        ### EXPLANATION: Yes, __call__ should only happen ONCE per backend instance.
        ### After this, the model uses the compiled split_gm directly.
        ###
        ### The caching works at a DIFFERENT level - it caches the compiled KERNELS
        ### (the output of Inductor), not the backend itself. So:
        ### - First run: __call__ runs, compiles everything, caches kernels to disk
        ### - Later runs: __call__ runs, but load() finds cached kernels (fast!)
        ### - Same session: compiled split_gm is used directly (no __call__)
        assert not self._called, "SGLangBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        ## What is gm? Why are we splitting the graph even if we may not have called for piecewise graph in the args
        ### gm = GraphModule (the traced FX graph)
        ###
        ### We ALWAYS split because piecewise compilation is the core strategy!
        ### The split_ops come from CompilationConfig, which defaults to including
        ### attention operations. Even if not explicitly requested, the config
        ### determines what to split on.
        ###
        ### If split_ops is empty, split_graph returns the original graph unsplit.
        self.split_gm, self.piecewise_graphs = split_graph(
            graph,
            self.compile_config.split_ops,
        )
        from torch._dynamo.utils import lazy_format_graph_code

        ## What this means?
        ### EXPLANATION: lazy_format_graph_code is a utility for debugging/logging.
        ### - It formats the FX graph as readable Python code
        ### - "lazy" means it only does the formatting if someone actually reads the log
        ### - depyf is a debugging tool that hooks into this to dump graphs to files
        ###
        ### This helps developers see:
        ### - "before split": The original traced graph
        ### - "after split": The graph with submodules for each piece
        ###
        ### Example "after split" might look like:
        ###   class GraphModule(torch.nn.Module):
        ###       def forward(self, x):
        ###           submod_0 = self.submod_0(x)
        ###           submod_1 = self.submod_1(submod_0)  # attention
        ###           submod_2 = self.submod_2(submod_1)
        ###           return submod_2
        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)

        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
            ### Only compile the capturable segments, NOT the split points
        ]

        ## Run the piecewise backend here
        ### CORRECT: This runs the interpreter which:
        ### 1. Traces through split_gm with fake tensors
        ### 2. For each capturable submodule, compiles it and installs a backend
        ### After this, split_gm's submodules are wrapped with CUDAPiecewiseBackend
        PiecewiseCompileInterpreter(
            self.split_gm,
            submod_names_to_compile,
            self.inductor_config,
            self.graph_pool,
            self.compile_config,
            self,
        ).run(*example_inputs)

        rank = torch.distributed.get_rank()

        if rank == 0:
            graph_path = os.path.join(
                local_cache_dir, f"computation_graph_{time.time()}.py"
            )
            if not os.path.exists(graph_path):
                # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
                # use `print_readable` because it can include submodules
                src = (
                    "from __future__ import annotations\nimport torch\n"
                    + self.split_gm.print_readable(print_output=False)
                )
                src = src.replace("<lambda>", "GraphModule")
                with open(graph_path, "w") as f:
                    f.write(src)
                ### Saves the final computation graph as readable Python for debugging

                rank0_log(f"Computation graph saved to {graph_path}")

        self._called = True
        return self.split_gm
        ### Return the split graph module. When called, it will:
        ### 1. Run submod_0 (CUDAPiecewiseBackend -> CUDA graph replay)
        ### 2. Run submod_1 (original attention)
        ### 3. Run submod_2 (CUDAPiecewiseBackend)
        ### ... and so on
```


```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         torch.compile(model, backend=SGLangBackend)         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TorchDynamo traces model.forward()                       │
│                    Produces: FX GraphModule (graph)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SGLangBackend.__call__(graph, example_inputs)            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    split_graph(graph, split_ops=["attention"])              │
│                                                                             │
│    Original:  embed → LN → attn → MLP → LN → attn → MLP → logits           │
│                                                                             │
│    Split:     [submod_0]  [submod_1]  [submod_2]  [submod_3]  [submod_4]    │
│               embed→LN     attn        MLP→LN      attn        MLP→logits  │
│               (capture)   (dynamic)   (capture)   (dynamic)    (capture)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PiecewiseCompileInterpreter.run()                        │
│                                                                             │
│    For each capturable submod (0, 2, 4):                                    │
│    1. Compile with Inductor for dynamic shapes (fallback)                   │
│    2. Wrap with CUDAPiecewiseBackend                                        │
│    3. Replace submod in split_gm                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Return: split_gm (with backends installed)               │
│                                                                             │
│    split_gm.forward(x):                                                     │
│        x = self.submod_0(x)  # CUDAPiecewiseBackend → CUDA graph            │
│        x = self.submod_1(x)  # Original attention (FlashAttention)          │
│        x = self.submod_2(x)  # CUDAPiecewiseBackend → CUDA graph            │
│        x = self.submod_3(x)  # Original attention                           │
│        x = self.submod_4(x)  # CUDAPiecewiseBackend → CUDA graph            │
│        return x                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
