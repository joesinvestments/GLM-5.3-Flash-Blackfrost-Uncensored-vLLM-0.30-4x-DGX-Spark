# Patches

The image is stock `vllm/vllm-openai:v0.30.0` plus 8 vLLM patches and 2 FlashInfer patches (and one optional vLLM patch). `image/build.sh` applies them and refuses to build unless every patched file matches `image/EXPECTED.sha256`, the checksums of the files in the image behind the published numbers.

**Where they come from.** Patches 01 to 08 port the GB10 changes from Tony's image and recipe ([tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)) onto stock vLLM v0.30.0. Every change was checked against upstream first: whatever v0.30.0 already ships was dropped, the rest was ported and re-verified. Two problems found while booting the port were fixed in patches 07 and 08. The two FlashInfer patches are Tony's. Patch 09 is new.

| Patch | File | What it does |
|---|---|---|
| 01-pdl-sm12x | `vllm/platforms/cuda.py` | Keeps programmatic dependent launch (PDL) off on SM12x, where it races on the KDA state kernels on GB10, and lists the SM90 sparse-MLA backend as an option on compute capability 12. |
| 02-dflash2-is-causal-config | `vllm/model_executor/models/qwen3_dflash.py` | Honors `dflash_config.is_causal`, which is where the GLM-5.3-Flash DFlash2 drafter declares causality. v0.30.0 only reads the top-level key. |
| 03-glm5next-kda-nvfp4 | `vllm/models/glm5next/nvidia/kda.py` | Stops stripping the quantization config when building the KDA (linear attention) layers, so their projections load as NVFP4. Modules that must stay BF16 are protected by the checkpoint's ignore list. |
| 04-glm5next-model-nvfp4-eagle3 | `vllm/models/glm5next/nvidia/model.py` | Builds the MLA attention with the model's quantization config (NVFP4 attention) and adds Eagle-3 style auxiliary hidden-state capture. |
| 05-kpool-indexer-fixes | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | Initializes the kpool top-k buffer with -1 instead of uninitialized memory, and keeps small-SM parts like GB10 (48 SMs, 99 KB shared memory) off the persistent top-k kernel, whose fallback needs 128 KB. |
| 06-kpool-compress-pool-len | `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | Bounds the kpool history expansion by the pool length, so out-of-range ids become -1 instead of garbage token ids. |
| 07-flashinfer-mla-sparse-sm90-sm12x | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` | Runs the SM90 sparse-MLA backend on SM12x: widened capability gate, the FlashInfer SM90 no-RoPE API required only for fp8 KV, the fa2 wrapper off SM90, and fp8 as the backend state's logical KV dtype (fixed during the port). |
| 08-glm5-drafter-kv-group | `vllm/v1/core/kv_cache_utils.py` | Lets the drafter's sliding-window KV layers coexist with GLM-5-Next's hybrid KV groups as one extra group with correct sizing and accounting (fixed during the port: the exact-fit drafter tensor uses the per-layer spec). |
| optional/09 | `vllm/models/glm5next/nvidia/kda.py`, `model.py` | Passes `--mamba-ssm-cache-dtype` through to the KDA recurrent state. Upstream v0.30.0 silently ignores the flag for GLM-5.3-Flash, so the state is always fp32. Without the flag, behavior is unchanged. See findings for why we don't use bf16 state. |
| fi-01 (Tony) | `flashinfer/mla/_core.py` | Allows the FP8 MLA path on compute capability 12 (the guard was SM90 only). |
| fi-02 (Tony) | `flashinfer/data/include/flashinfer/attention/mla.cuh` | Clamps the effective CTA KV tile to 32 for the FP8 MLA kernel. |

**Dropped because v0.30.0 already has them:** forcing the V2 model runner for DFlash2 drafts, vocab-parallel top-k logits, the DFlash2 drafter model and its registry entry, DFlash2 speculator routing, and the parameterized draft-logits buffer.

**Serving config these patches assume** (see `launch/launch_node.sh`): sparse MLA through the SM90 FlashInfer backend with fa2 and fp8 KV cache, Marlin for the NVFP4 MoE, DFlash2 drafting 7 tokens, `FULL_AND_PIECEWISE` CUDA graphs, and `VLLM_MARLIN_USE_ATOMIC_ADD=1`.
