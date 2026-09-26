# GLM-5.3-Flash Blackfrost Uncensored on 4x DGX Spark

**A reproducible build for serving Blackfrost's uncensored GLM-5.3-Flash ([DERISKED](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4), NVFP4) across four NVIDIA DGX Spark (GB10) nodes at tensor parallel 4, with DFlash2 speculative decoding.**

Since 2026-09-26 our production runs the **vLLM main nightly** (commit `7f1a5398e9`) with our 18-patch series, the **Humming** MoE backend and **Z.ai's official chat template**. The earlier build on the vLLM v0.30.0 release is still here and still works. Both builds start from stock vLLM images, copy in patched files only, and refuse to build unless every patched file matches the checksums of the image they reproduce: today's production for the main build, the previous production (now the rollback) for the v0.30 build. Everything was measured on our own cluster, including the experiments that did not pay off.

| | Main build (production) | v0.30 build (kept) |
|---|---|---|
| Base image | `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423` | `vllm/vllm-openai:v0.30.0` |
| vLLM patches | 18, applied in the order of `patches/vllm/series` | 13, plus an optional one (09) |
| FlashInfer patches | Tony's 2 GB10 fixes | the same 2 |
| MoE backend | Humming | Marlin |
| Optional NCCL fence fix | NCCL 2.30.7 + [NVIDIA/nccl#2393](https://github.com/NVIDIA/nccl/pull/2393) | NCCL 2.29.7 + #2393 |
| Build and launch | `image-main/build.sh`, `launch/launch_node.sh` | `image/build.sh`, `launch/launch_node_v030.sh` |

Everything else is identical in the two launchers: our NVFP4 checkpoint, incoai's DFlash2 drafter with 8-bit projections, block verification, draft length by batch size, `disable_eagle_block_drop`, a 36 GiB fp8 KV cache, 2,304-token blocks and the chat template.

## Why vLLM main plus Humming

vLLM main is where upstream development happens. Fixes land there first, and staying on a release means porting each one back by hand, as we did for three long-context fixes on 0.30 (see [The v0.30 build](#the-v030-build)). The 7f1a nightly already has one of those fixes and we carry the other two verbatim. [Humming](https://github.com/vllm-project/humming) is a vLLM project library of "JIT-compiled GPU GEMM kernels for quantized inference", newer than the Marlin kernels the v0.30 build uses.

We chose this path knowing what it costs, measured on the long agent-session load below:

- **Throughput: about 5 to 7% lower.** vLLM main with our series ran at 0.955x of v0.30, and with Humming 0.933x (geometric mean of aggregate tok/s over 1, 2, 4 and 8 sessions). Humming against vLLM main alone: 0.977x overall, but 0.93x, 1.18x, 0.88x and 0.95x per level. v0.30 measured against itself on two more boots read 0.946x and 1.003x, so a few percent is inside run-to-run noise. The cost is real but small.
- **Time to first token: about the same,** 1.019x (v0.30 against itself: 0.989x and 0.992x).
- **Quality: equal or better** on every probe we run (table below).

## Results (2026-09-26)

**Load:** long agent sessions (about 19K-token prompts) at 1, 2, 4 and 8 concurrent sessions, every request with `reasoning_effort` max (as our agent clients send it), replies capped at 768 tokens, 2 sweeps per stack. v0.30 is pooled over 3 boots.

Aggregate tok/s and time to first token:

| Sessions | v0.30 | vLLM main + our series | **Final stack** (+ Humming) |
|---|---|---|---|
| 1 | 38.4 tok/s, 1.72 s | 36.2 tok/s, 1.73 s | 33.7 tok/s, 1.71 s |
| 2 | 48.5 tok/s, 1.78 s | 41.0 tok/s, 1.74 s | 48.3 tok/s, 1.76 s |
| 4 | 58.2 tok/s, 2.60 s | 63.7 tok/s, 2.70 s | 55.8 tok/s, 2.78 s |
| 8 | 69.7 tok/s, 3.81 s | 66.5 tok/s, 3.77 s | 63.1 tok/s, 3.90 s |
| vs v0.30, geometric mean | | 0.955x tok/s, 1.002x TTFT | 0.933x tok/s, 1.019x TTFT |
| Prefix-cache hit rate | 0.827 | 0.827 | 0.827 |
| Draft acceptance | 0.342 | 0.334 | 0.337 |

Quality:

| Probe | v0.30 | vLLM main + our series | Final stack |
|---|---|---|---|
| NLL/token, 40 WikiText-2 chunks | 1.26625, 1.26637, 1.26686 (3 boots) | 1.26750 | 1.26630 |
| NLL/token, 2 WikiText-2 documents of about 11K tokens | 1.57878, 1.11404 | 1.57540, 1.11261 | 1.57775, 1.09292 |
| GSM8K, 250 problems, thinking off | 98.4% | 98.4% | 98.4% |
| Cached tokens when 6 long sessions resume | 18,432 / 18,432 / 18,432 / 16,128 / 16,128 / 18,432 | the same | the same |

- Uncensored gate (65 adult prompts in 12 categories): PASS for vLLM main and for the final stack.
- The checkpoint A/B of 2026-09-24 (below) read 99.2% on GSM8K for this checkpoint; this run read 98.4% on all three stacks.
- The results above were measured on the main build before its last two patches (u58834, u58021) were added on 2026-09-26. Neither runs in this configuration, so they describe what serves.

### Short prompts on the main build (2026-09-26)

The same benchmark as the v0.30 table further down, run on production (the main build) on 2026-09-26: `tools/ab_harness.py` 3 times (temperature 0, thinking off, max 400 tokens; median of 3) and `tools/agent_sessions.py` 3 sweeps (3-turn tool loops with 1 to 3K-token prompts, half the sessions thinking, default sampling; mean of 3). No other traffic reached the server during either run.

| Load | Main build (production) | Range | v0.30 build (2026-09-25) | First release (2026-09-23) |
|---|---|---|---|---|
| 1 user, simple output (counting) | 128 tok/s | 128 to 128 | 133 tok/s | 128 tok/s |
| 1 user, JSON | 87 tok/s | 85 to 88 | 89 tok/s | 92 tok/s |
| 1 user, code | 82 tok/s | 79 to 84 | 84 tok/s | 88 tok/s |
| 1 user, step-by-step math | 74 tok/s | 68 to 76 | 76 tok/s | 72 tok/s |
| 1 user, explanation | 48 tok/s | 46 to 51 | 50 tok/s | 50 tok/s |
| 1 user, creative prose | 38 tok/s | 35 to 38 | 38 tok/s | 38 tok/s |
| 8 users, aggregate | 257 tok/s | 253 to 269 | 282 tok/s | 287 tok/s |
| 32 users, aggregate | 572 tok/s | 572 to 579 | 564 tok/s | 591 tok/s |
| 1 agent session | 41 tok/s | 38 to 45 | 42 tok/s | 44 tok/s |
| 2 agent sessions | 54 tok/s | 47 to 64 | 47 tok/s | 52 tok/s |
| 4 agent sessions | 71 tok/s | 66 to 76 | 66 tok/s | 69 tok/s |
| 8 agent sessions | 76 tok/s | 66 to 82 | 91 tok/s | 83 tok/s |

- Single-user speed is 2 to 5% below the v0.30 build on most prompts, and 32 users are level. The clearest loss is 8 concurrent agent sessions: 76 against 91 tok/s, with ranges that do not overlap (66 to 82 against 89 to 93). That matches the long-session results above, where 8 sessions was also the weakest level (0.905x).
- Draft acceptance on the one-shot prompts: 0.404 (3.83 tokens per step), the same as the v0.30 build (0.405).
- These are 3 runs against 3 runs on different days, so only gaps larger than the ranges mean much.

## What's here

```
image-main/  the production build: build.sh, patches/vllm/series (18 vLLM patches), 2 FlashInfer patches,
             EXPECTED.sha256, nccl/ (optional NCCL 2.30.7 fence fix)
image/       the v0.30 build: build.sh, 13 vLLM + 2 FlashInfer patches + optional 09, EXPECTED.sha256, nccl/
launch/      launch_node.sh (main build) and launch_node_v030.sh (v0.30 build): the exact serving flags, one rank per node
template/    chat_template_zai0907_optout.jinja: Z.ai's official 09-07 template plus one opt-out line
tools/       ab_harness.py (speed + acceptance), agent_sessions.py (1 to 8 concurrent agent sessions),
             quality_eval.py, analyze_trace.py (profiler breakdown)
tools/lab/   lab_run.sh: a safe way to run experiments (or an AI agent) on a cluster that is also serving
docs/        patches.md (what each patch does and where it came from), findings.md (profile, experiments, lessons)
```

## Quick start

**1. Build the image on each node.** You need the base image locally (about 20 GB):
```bash
docker pull vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423
cd image-main && ./build.sh      # builds glm53-flash-gb10:main-7f1a
```
The script copies the target files out of the base image, applies the series in order plus the FlashInfer patches, and refuses to build unless every result matches `EXPECTED.sha256`, the checksums of the files in the image we serve. The build only copies files; no container runs. To include the NCCL fence fix, run `nccl/build_nccl.sh` first (about 12 minutes on a Spark) and then `NCCL_LIB=nccl/libnccl.so.2 ./build.sh`.

*Release-based alternative:* `cd image && ./build.sh` builds `glm53-flash-gb10:v0.30.0` from stock `vllm/vllm-openai:v0.30.0` (Marlin MoE, `WITH_PATCH_09=1` adds the optional patch). Launch it with `launch/launch_node_v030.sh` in step 4.

**2. Get the weights onto every node** (not redistributed here; follow each license):
- Checkpoint: [Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4), converted to NVFP4 attention with Tony's recipe in [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) (`runs/2026-09-21-blackfrost-derisked/`). The numbers above use our requantization of the same weights from Blackfrost's BF16 (same tensors, same layout, a drop-in; see `docs/findings.md`).
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at revision `7d74cdd881ed7e32c31175984a67823127b66cfe`, in `$HF_ROOT/hub/glm53-flash-dflash2/`. Patch 10 packs its projections to 8 bits when the server loads it (`VLLM_DRAFTER_W8A16=1` in the launcher); the files on disk are unchanged.

**3. Put the chat template next to the checkpoint** on every node:
```bash
cp template/chat_template_zai0907_optout.jinja "$HF_ROOT/hub/<checkpoint dir>/"
```
Both launchers pass it with `--chat-template` and refuse to start without it. It is Z.ai's official 09-07 GLM-5.3-Flash template plus one explicit opt-out line. There is no server default that turns thinking off: every request reasons unless it sends `"chat_template_kwargs": {"enable_thinking": false}`. With thinking on, it renders byte for byte like the official template.

**4. Launch, workers first, head last:**
```bash
export NODES="<ip0> <ip1> <ip2> <ip3>"   # rail IPs, rank 0 (head) first
export MODEL=<checkpoint dir under $HF_ROOT/hub>
./launch/launch_node.sh 3   # on node 3, then 2 and 1
./launch/launch_node.sh 0   # on the head node
```
Defaults assume the Spark's QSFP port shows up as `enp1s0f0np0` / `enP2p1s0f0np0` (RDMA devices `rocep1s0f0` / `roceP2p1s0f0`) and RoCE GID index 3; see the variables at the top of the script. Point `CACHE` at a fresh directory for each image build (it holds the JIT caches). We run every rank under [oomwrap](https://github.com/osolmaz/oomwrap) with `tools/lab/oomwrap_supervise.sh`.

**5. Verify with a real completion,** not just `/health`. Thinking is on by default, so turn it off for a quick check:
```bash
curl -s http://HEAD:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":16,"messages":[{"role":"user","content":"Say OK"}],
       "chat_template_kwargs":{"enable_thinking":false}}'
```

**Memory:** each Spark runs earlyoom (SIGTERM at 4% free, about 4.9 GB). The 36 GiB KV cache in the launchers keeps the head node clear of that line; 38 GiB got the head node's vLLM worker killed once, 44 GiB twice. See `docs/findings.md`.

## The main build's patches

Details are in [docs/patches.md](docs/patches.md).

| Group | Patches | What they do |
|---|---|---|
| Upstream vLLM | u57632, u58704, u58454, u58250, u58834, u58021 | [#57632](https://github.com/vllm-project/vllm/pull/57632) cherry-picked (DFlash context KV in the draft CUDA graph); [#58704](https://github.com/vllm-project/vllm/pull/58704) and [#58454](https://github.com/vllm-project/vllm/pull/58454) verbatim (long-context fixes the 7f1a nightly does not have yet); vLLM's own revert #58250 of the fused DFlash2 grouped convolution (#55960); two open fixes carried verbatim to stay aligned with upstream, both inactive in this configuration: [#58834](https://github.com/vllm-project/vllm/pull/58834) (GLM-5.3-Flash decoder layout under sequence-parallel MoE, non-mHC branch) and [#58021](https://github.com/vllm-project/vllm/pull/58021) (rejects a `prefix_match_unit` that a single KV cache group cannot honor) |
| GB10 ports (Tony's lineage) | 01 to 08 | PDL off on SM12x; DFlash2 causality read from `dflash_config`; NVFP4 KDA and MLA layers with Eagle-3 auxiliary hidden states; kpool indexer fixes; FlashInfer SM90 sparse MLA on SM12x; the drafter's KV group (fixed 2026-09-25, see below) |
| jnardiello's experiments, ported | 10, p2 | 8-bit drafter projections, a port of his E22b (`VLLM_DRAFTER_W8A16=1`); 8-bit MLA fused qkv_a, a port of his E21 idea (off unless `VLLM_E21_BF16_RESIDUE_W8A16=1`) |
| Ours | 09, p4 | `--mamba-ssm-cache-dtype` reaches the KDA state; keeps our NVFP4 attention scales through an upstream fp8 attention helper that swallowed them |
| FlashInfer (Tony) | fi-01, fi-02 | FP8 MLA allowed on compute capability 12; CTA KV tile clamped to 32 |

## Key findings

More detail and numbers are in [docs/findings.md](docs/findings.md).

- **Resumed sessions and our port of patch 08.** On 2026-09-25 our vLLM main build lost prefix-cache reuse on resumed agent sessions (the first time we ran the resumed long-session check on it): cached tokens on six resumes were 0 / 0 / 0 / 13,824 / 13,824 / 0 instead of production's 18,432 / 18,432 / 18,432 / 16,128 / 16,128 / 18,432. Our port of patch 08 marked the drafter's KV group as an EAGLE group unconditionally. With `disable_eagle_block_drop`, vLLM then wrote the KDA recurrent state at the last full block but kept the state one block lower, where nothing had been written. Upstream only flags groups when the block drop is on, so the fix gates the flag on `use_eagle_block_drop()`. A CPU replay of vLLM's real scheduler and KV manager reproduced the cluster's exact numbers before the fix and production's after; on hardware the fixed build matched production exactly.
- **Short-turn caching: built, tested, not shipped.** vLLM main has `--prefix-match-unit`, but on this stack it did nothing: the DFlash2 drafter's sliding-window KV group has no fine-grained lookup, so vLLM switches fine-grained hits off for the whole model. We added that lookup (together with upstream [#58368](https://github.com/vllm-project/vllm/pull/58368) and [#58434](https://github.com/vllm-project/vllm/pull/58434)). On hardware, a follow-up turn resumed exactly at the previous prompt's last 256-token boundary (after 1,800 tokens it reused 1,792; after 3,000, 2,816; after 19,000, 18,944). Short agent turns (about 1.5K to 2.5K-token prompts) went from 0% to 59% prefix-cache hits and 0.835x time to first token; long sessions ran 1.04x tok/s. Quality: NLL/token 1.26564 on the 40 chunks, long documents 1.57903 and 1.11577 (v0.30: 1.57878 and 1.11404), GSM8K 98.4%, uncensored gate PASS. It failed one pre-registered correctness check: after a resume in the middle of a block, the first token's logprobs drifted from a fresh computation by up to 0.237, against 0.089 for ordinary full-block resumes (the bar was 2x the control). The 0.237 is one of eight resumes; the other seven stayed within the control's range. It stays out until that drift is understood.
- **Test long context, not just short passages.** Three upstream bugs in GLM-5.3-Flash's attention cache only showed past 2,048 tokens. Our short-chunk quality eval passed all along; a two-document, 11K-token NLL probe found them.
- **Chat template.** The v0.30 build served Tony's multimodal template, older than Z.ai's official 09-07 one, with a server default of thinking off. Our agent clients always request reasoning, and vLLM turns thinking on when `reasoning_effort` is sent. Both launchers now use the official template plus one opt-out line.
- **No drafter we tested beats incoai's DFlash2 on this model.** The model's own MTP head: 0.96x tok/s, 1.05x time to first token on long agent sessions. Always drafting 7 tokens instead of the per-batch-size schedule: 0.97x on the long-session load against one production run, so no evidence it beats the schedule. RedHat's DSpark preview drafter was not measured: on this hybrid stack its attention layers made vLLM pick 4,608-token cache blocks and pad the recurrent-state pages 99%, and our pre-registered boot check stopped it (it needs a block-size fix). canada-quant's DFlash2-G was not tested: its serving patches are not public, it needs new code for its full-attention layers, and its own card shows lower throughput than incoai's drafter. What helped was making the drafter cheaper (8-bit projections) and drafting less as the batch grows.
- **Where decode time goes** (profiled on the v0.30 build with Marlin MoE). At 1 user: MoE experts 45%, BF16 GEMMs 19%, NCCL 13%, GPU idle 6%. At 32 users: MoE 47%, the KDA linear-attention kernel 19%, NCCL 12%, idle 1%. The MoE is at the memory-bandwidth limit.
- **Component speedups did not carry into serving.** A graph-capturable RoCE all-reduce 2 to 4x faster than NCCL, and a 2x faster KDA kernel from bf16 state, each moved end-to-end throughput by only about 2%. Judge changes by a serving A/B.
- **Agent sessions and the prefix cache.** The KDA state is only saved at 2,304-token block boundaries, so short agent turns (under one block) get no prefix-cache hits. Long sessions do, and keeping the last cached block on resume (`disable_eagle_block_drop`) cut time to first token by a third. vLLM main's `--prefix-match-unit` can resume short turns at the prompt's last 256-token boundary, but on this stack it does nothing without a fix for the drafter's KV group; we built one and did not ship it (see the short-turn caching work above).
- **A vLLM bug:** `--mamba-ssm-cache-dtype` is silently ignored for GLM-5.3-Flash. Patch 09 fixes it (applied in the main build, optional in the v0.30 build).
- **A network trap on Spark clusters:** a netplan `match: {}` let NetworkManager bring the rail address up on the wrong QSFP function after a network blip, while `/health` stayed green.

## The v0.30 build

The first release (2026-09-23) and the 2026-09-25 changes (production for a day, never published on their own) ran on the **vLLM v0.30.0 release**: stock `vllm/vllm-openai:v0.30.0` plus 13 vLLM patches and 2 FlashInfer patches. It grew out of Tony's build, which runs on the launch-day vLLM image, and ported the GB10 fixes onto the release. It served production until 2026-09-26 and stays here as the release-based alternative. Patch numbers in this section refer to `image/patches/`.

### Update 2026-09-25: long-context fixes, a better checkpoint, faster agent sessions

Each change was measured against the build before it, on the load it targets, with the pass rule written before the first run:

| Change | Result | Measured with |
|---|---|---|
| **Three long-context correctness fixes** (patches 12 to 14) | NLL/token on two long documents 1.578 and 1.118, down from 1.648 and 1.225 (4.3% and 8.8% lower). Speed unchanged: 1.04x tok/s, 0.99x time to first token | two WikiText-2 documents of about 11K tokens each (prompt logprobs); long agent sessions at 1, 2, 4 and 8 |
| **Our own NVFP4 checkpoint**, requantized from Blackfrost's BF16 with an MSE scale search | NLL/token 1.26587 vs 1.29080 (paired difference -0.025, plus or minus 0.0075); GSM8K 99.2% vs 98.4%. The same agent tasks finished in 0.80x the time (fewer answers ran into the length limit: 1 vs 7), at the same tok/s | 40 WikiText-2 chunks and 250 GSM8K problems; agent sessions with both checkpoints in one session |
| **8-bit drafter projections** (patch 10) | +8.6% (95% interval +5.0% to +12.6%); 8 sessions 84.5 vs 71.0 tok/s; acceptance unchanged | agent sessions at 1, 2, 4 and 8, paired by sweep |
| **Keep the cached block when an agent session resumes** (`disable_eagle_block_drop`) | time to first token 0.68x (1 session 2.43 to 1.63 s, 8 sessions 5.7 to 3.9 s); prefix-cache hit rate 70% to 83% | long agent sessions (about 19K-token prompts) at 1, 2, 4 and 8, production measured before and after |
| **Block verification and draft length by batch size** (7 tokens at 1 request, 5 at 2 to 3, 4 at 4 or more) | +4.4% (95% interval -0.4% to +9.7%, so within noise) | agent sessions at 1, 2, 4 and 8 |
| **NCCL fence fix** (optional, `image/nccl/`) | fixes a rare RoCE hang on Arm ([NVIDIA/nccl#2334](https://github.com/NVIDIA/nccl/issues/2334)); not a speed change | in production since 2026-09-23 |

**The long-context bugs are in stock vLLM 0.30** and hit every GLM-5.3-Flash request past 2,048 tokens of context, which is every agent turn:
- [vllm#58704](https://github.com/vllm-project/vllm/pull/58704): the SM90 sparse MLA backend took the index pool size from the KV cache spec (1) instead of the model (4), so the current token and up to 2 before it went unread on 3 of 4 positions in all 11 MLA layers. The reporter tied it to failed tool calls at 34K to 42K tokens.
- [vllm#58454](https://github.com/vllm-project/vllm/pull/58454): with speculative decoding, rejected drafts overwrote the pool's tail ring and corrupted pool keys built during decode. The PR's repro in our serving image: 106 of 128 key bytes wrong.
- [vllm#57477](https://github.com/vllm-project/vllm/pull/57477): the prefill seed kernel addressed a padded cache layout as if it were dense, so every prefill wrote into the indexer cache of the lowest block ids, which usually hold the oldest cached prompts.

vLLM merged all three after v0.30.0 was cut (#57477 on 2026-09-20, the other two on 2026-09-25). Patch 12 is #58704's one-line change plus a comment; 13 and 14 are ports to 0.30, checked against the merged code. Kernel tests for the ports fail on the unfixed image and pass on this one (46 of 46 and 6 of 6 checks). A short-passage quality eval (chunks under 2,048 tokens) cannot see any of the three: the long-document probe is what caught them. In the main build, #57477 is already in the 7f1a nightly and the other two are carried verbatim.

### What the v0.30 build added over the launch-day image

Measured 2026-09-22 against the launch-day-image build, same checkpoint and serving setup:

| | Launch-day image | v0.30.0 build |
|---|---|---|
| Prompt processing | 1,761 to 2,016 tok/s | **2,211 to 2,496 tok/s** (about 25% faster) |
| Launch to serving | about 10 minutes | **about 4 minutes** (InstantTensor weight loading) |
| Follow-up on a long prompt, time to first token | baseline | **about 12x faster** (0.30 prefix-caches the hybrid KDA model) |

### Short prompts on the v0.30 build (2026-09-25)

The 2026-09-25 build vs the first release (2026-09-23). `tools/ab_harness.py` (temperature 0, thinking off, max 400 tokens, 3 runs now vs 12 then) and `tools/agent_sessions.py` (3-turn tool loops with 1 to 3K-token prompts, half the sessions thinking, default sampling, 3 sweeps now vs 4 then):

| Load | 2026-09-25 build | Range | First release |
|---|---|---|---|
| 1 user, simple output (counting) | 133 tok/s | 131 to 133 | 128 tok/s |
| 1 user, JSON | 89 tok/s | 88 to 90 | 92 tok/s |
| 1 user, code | 84 tok/s | 84 to 86 | 88 tok/s |
| 1 user, step-by-step math | 76 tok/s | 70 to 82 | 72 tok/s |
| 1 user, explanation | 50 tok/s | 49 to 52 | 50 tok/s |
| 1 user, creative prose | 38 tok/s | 37 to 40 | 38 tok/s |
| 8 users, aggregate | 282 tok/s | 250 to 298 | 287 tok/s |
| 32 users, aggregate | 564 tok/s | 563 to 579 | 591 tok/s |
| 1 agent session | 42 tok/s | 36 to 46 | 44 tok/s |
| 2 agent sessions | 47 tok/s | 44 to 50 | 52 tok/s |
| 4 agent sessions | 66 tok/s | 59 to 75 | 69 tok/s |
| 8 agent sessions | 91 tok/s | 89 to 93 | 83 tok/s |

- Speculative decoding: 0.405 of drafted tokens accepted on the one-shot prompts (3.83 tokens per step), 0.391 before.
- **Read this table as "no change within noise."** None of the 2026-09-25 changes target short one-shot prompts, and the agent-session load swings up to 15% between sweeps and between days (on 2026-09-24 the same load read 71 tok/s at 8 sessions). The per-change gains above come from paired A/Bs run in one session each. One real cost shows up here: at 4 or more requests the build drafts 4 tokens instead of 7, which was measured on agent traffic at default sampling; on temperature-0 one-shot prompts at 32 users it costs about 5%.
- The agent sessions finished the same tasks sooner at 1 and 8 sessions (42 vs 66 s and 48 vs 69 s) and later at 2 (82 vs 53 s), mostly because the answers were shorter or longer; the new checkpoint's paired test is the clean measurement (0.80x the time for the same tasks).

## Credits

- **Tony ([@2WildTech](https://x.com/2WildTech), [tonyd2wild](https://github.com/tonyd2wild))**: the GB10 patch set and image that patches 01 to 08 are ported from, the two FlashInfer FP8 MLA fixes, and the NVFP4-attention conversion recipe. Both builds start from stock vLLM images; his work reaches them as patches.
- **[Blackfrost](https://huggingface.co/Blackfrost-AI) ([@Blackfrost_AI](https://x.com/Blackfrost_AI))**: GLM-5.3-Flash-DERISKED (MIT), in NVFP4 and BF16. The checkpoint served here requantizes their weights; per their model card, Blackfrost has not evaluated modified versions. **[Z.ai](https://huggingface.co/zai-org) ([@Zai_org](https://x.com/Zai_org))**: GLM-5.3-Flash (MIT) and the official chat template.
- **[incoai](https://huggingface.co/incoai)**: the GLM-5.3-Flash DFlash2 drafter (CC BY-NC-ND 4.0), built on DFlash from [Z Lab](https://github.com/z-lab/dflash) ([@zhijianliu_](https://x.com/zhijianliu_)).
- **[Humming](https://github.com/vllm-project/humming)** ([Jinzhen Lin](https://github.com/jinzhen-lin), [Julian Huang](https://github.com/huangzhilin-hzl), [Misha Goin](https://github.com/mgoin) ([@mgoin_](https://x.com/mgoin_)) and the Humming contributors, Apache-2.0): the MoE backend in the main build.
- **vLLM contributors in the shipped stack:** [Matt Mastracci](https://github.com/mmastrac) ([@mmastrac](https://x.com/mmastrac); #58704, #58454), [JaredforReal](https://github.com/JaredforReal) (#57477, now in vLLM main) and [Juntian777](https://github.com/Juntian777) (#57632), [shiweijiezero](https://github.com/shiweijiezero) (#58834) and [QHarshil](https://github.com/QHarshil) (#58021). We also carry vLLM's own revert #58250 of [mgoin](https://github.com/mgoin)'s fused DFlash2 grouped convolution (#55960). Tested in the short-turn caching work, not shipped: [netanel-haber](https://github.com/netanel-haber) (#58368) and [njhill](https://github.com/njhill) (#58434).
- **NCCL:** [Stanislav Bardyuk](https://github.com/kodlan) wrote [NVIDIA/nccl#2393](https://github.com/NVIDIA/nccl/pull/2393), the fix for the RoCE deadlock we reported in [#2334](https://github.com/NVIDIA/nccl/issues/2334); [Rami Nudelman](https://github.com/raminudelman) at NVIDIA brought it to us.
- **[Jacopo Nardiello](https://github.com/jnardiello) ([@jnardiello](https://x.com/jnardiello))** ([GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)): the 8-bit drafter (his E22b) that patch 10 ports, and the E21 8-bit residue idea behind patch p2 (off by default).
- **[vLLM](https://github.com/vllm-project/vllm) ([@vllm_project](https://x.com/vllm_project))**, **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** and **[LLM Compressor](https://github.com/vllm-project/llm-compressor)** (Apache-2.0).
- **[local-inference-lab](https://github.com/local-inference-lab)**: b12x and RoCEnante, tested here. **[modal-labs](https://huggingface.co/modal-labs)**: the DFlash drafter we compared against.
- **[oomwrap](https://github.com/osolmaz/oomwrap)** by [Onur Solmaz](https://github.com/osolmaz) ([@onusoz](https://x.com/onusoz)): memory-pressure supervision for every launch.

## License

The scripts, tools and patches in this repo are Apache-2.0 (see `LICENSE`); the patches modify Apache-2.0 vLLM and FlashInfer code and BSD-3-Clause NCCL code. Humming (Apache-2.0) is not included here; the main build selects it with `--moe-backend humming`. `template/` is Z.ai's official chat template with one added line and keeps Z.ai's license. No model weights are included. Each model has its own license: the incoai drafter is non-commercial and no-derivatives.
