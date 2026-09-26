# Findings

Everything below was measured on 4x DGX Spark (GB10, 128 GB unified memory each) with ConnectX-7 RoCE through a switch. Sections 1 to 6 were measured on the v0.30 build (`image/`) as it stood on each test's date (`launch/launch_node_v030.sh` is its final serving config), except the experiments-table rows that name vLLM main. The 2026-09-26 section compares that build with the vLLM main build production runs now (`image-main/`, served with `launch/launch_node.sh`).

Patch numbers in sections 1 to 6 are the v0.30 build's (`image/patches/`). In `image-main/`, patches 09 and 10 keep their numbers, patches 12 and 13 are carried as `u58704` and `u58454`, patch 14 (vllm#57477) is already in vLLM main, and patch 11 (the unused 8-bit lm_head) is not carried.

Unless a section says otherwise, serving numbers in sections 1 to 6 come from `tools/ab_harness.py`: temperature 0, thinking off, max 400 tokens, same prompts every run. Between identical runs of the same config on that load, aggregate throughput moves 2 to 4% and acceptance about 0.02, so smaller differences are noise. The long agent-session load used on 2026-09-26 moves more; see "Benchmarking: production against itself" below.

## 2026-09-26: vLLM main, Humming, and the port fix

### What production runs now

- **vLLM main**, nightly commit `7f1a5398e9` (it reports itself as 0.30.1rc1.dev48), built as a copy-only layer over the official nightly image by `image-main/build.sh`.
- **Our 18-patch series** (16 vLLM files, 18 with the two FlashInfer fixes; order in `image-main/patches/vllm/series`, details in `docs/patches.md`). Next to the patches from the v0.30 build it carries: a cherry-pick of [vllm#57632](https://github.com/vllm-project/vllm/pull/57632) (DFlash context K/V in the draft CUDA graph); [vllm#58704](https://github.com/vllm-project/vllm/pull/58704) and [vllm#58454](https://github.com/vllm-project/vllm/pull/58454) verbatim; vLLM's own revert [#58250](https://github.com/vllm-project/vllm/pull/58250) of the fused DFlash2 grouped convolution ([#55960](https://github.com/vllm-project/vllm/pull/55960)); two open upstream fixes, vllm[#58834](https://github.com/vllm-project/vllm/pull/58834) and vllm[#58021](https://github.com/vllm-project/vllm/pull/58021), carried verbatim since 2026-09-26 and inactive in this configuration (sequence-parallel MoE is off, and the model has several KV cache groups), so the measurements below, taken before they were added, still describe what runs; `p4`, which keeps our NVFP4 attention scales that an upstream fp8 attention helper otherwise swallowed; `p2`, an 8-bit MLA fused qkv_a projection ported from jnardiello's E21 idea, off unless `VLLM_E21_BF16_RESIDUE_W8A16=1`; and patch 08 with the fix described below. The two FlashInfer GB10 fixes come along unchanged. Production also carries NCCL 2.30.7 with the fence fix from [NVIDIA/nccl#2393](https://github.com/NVIDIA/nccl/pull/2393) (optional in the build, `image-main/nccl/`).
- **Humming 0.1.16 as the MoE backend** (`--moe-backend humming`; [vllm-project/humming](https://github.com/vllm-project/humming), JIT-compiled GEMM kernels for quantized models, Apache-2.0). The v0.30 build uses Marlin.
- **Everything else is the v0.30 build's serving setup:** our NVFP4 checkpoint (MSE requantization of Blackfrost's BF16), incoai's DFlash2 drafter at `7d74cdd` with 8-bit projections, block verification, draft length by batch size (7 tokens at 1 request, 5 at 2 to 3, 4 at 4 or more), `disable_eagle_block_drop`, a 36 GiB fp8 KV cache, 2,304-token blocks, 8,192 batched tokens, reasoning parser `glm45`, tool parser `glm47`.
- **A new chat template:** Z.ai's official 09-07 template with one opt-out line (see "Chat template and the thinking default").

The v0.30 build stays in the repo (`image/`, `launch/launch_node_v030.sh`). On the load below it is still the faster of the two.

### Moving to vLLM main and Humming: what it cost

Load for the serving numbers below (tok/s, TTFT, hit rate, acceptance): long agent sessions (prompts of about 19K tokens) at 1, 2, 4 and 8 concurrent sessions, every request sending `reasoning_effort: max` (what our agent clients send), replies capped at 768 tokens, 2 sweeps per arm. The v0.30 baseline is pooled over 3 boots. Aggregate tok/s and time to first token (TTFT):

| Stack | 1 session | 2 sessions | 4 sessions | 8 sessions | vs v0.30, geometric mean over the 4 levels |
|---|---|---|---|---|---|
| v0.30 build (production until 2026-09-26, 3 boots pooled) | 38.4 tok/s, 1.72 s | 48.5, 1.78 s | 58.2, 2.60 s | 69.7, 3.81 s | 1 |
| vLLM main + our series | 36.2, 1.73 s | 41.0, 1.74 s | 63.7, 2.70 s | 66.5, 3.77 s | 0.955x tok/s, 1.002x TTFT |
| vLLM main + our series + Humming (production now) | 33.7, 1.71 s | 48.3, 1.76 s | 55.8, 2.78 s | 63.1, 3.90 s | 0.933x tok/s, 1.019x TTFT |

- **Humming against vLLM main without it:** 0.977x tok/s (0.93, 1.18, 0.88 and 0.95 at 1, 2, 4 and 8 sessions) and 1.018x TTFT.
- **The throughput cost is real but small: about 5 to 7%.** vLLM main alone read 0.955x and the final stack 0.933x. Single levels swing both ways (vLLM main alone read 0.84x at 2 sessions and 1.09x at 4), which is why we read the geometric mean. Production against itself on this load read as low as 0.946x (see "Benchmarking" below), so vLLM main alone is inside the noise and the final stack sits just beyond it.
- **Prefix cache and drafting did not change.** Hit rate 0.827 on all three stacks. Draft acceptance 0.342 (v0.30), 0.334 (vLLM main) and 0.337 (with Humming); on this load acceptance is lower than on the one-shot prompts of section 1.
- **Cache resume on six resumed long sessions** (cached tokens per session): 18,432, 18,432, 18,432, 16,128, 16,128 and 18,432 on all three stacks.

Quality, same builds:

| Probe | v0.30 (3 boots) | vLLM main + our series | + Humming (production now) |
|---|---|---|---|
| NLL/token, 40 WikiText-2 chunks | 1.26625, 1.26637, 1.26686 | 1.26750 | 1.26630 |
| NLL/token, two WikiText-2 documents of about 11K tokens | 1.57878 and 1.11404 | 1.57540 and 1.11261 | 1.57775 and 1.09292 |
| GSM8K, 250 problems, thinking off | 98.4% | 98.4% | 98.4% |

The uncensored check (65 adult prompts across 12 categories) passed on both vLLM main builds.

Production runs the Humming stack by choice, with the cost above stated. On vLLM main, #57477 (patch 14 in the v0.30 build) is already upstream, and `--prefix-match-unit` exists, which the short-turn caching test below builds on.

### Resumed sessions lost their cache: cause and fix

On 2026-09-25 our vLLM main build lost prefix-cache reuse on resumed agent sessions (the first time we ran the resumed long-session check on it). The six resumed long sessions above got 0, 0, 0, 13,824, 13,824 and 0 cached tokens where production got 18,432, 18,432, 18,432, 16,128, 16,128 and 18,432. We logged it as a cost of rebasing (the "Rebasing onto vLLM main" row in section 2) and stayed on 0.30. The cause was in our port of patch 08, not in upstream vLLM.

**Cause.** Patch 08 lets the DFlash2 drafter's sliding-window KV layers live next to GLM-5.3-Flash's hybrid KV groups. Our port of it to vLLM main marked the drafter's group as an EAGLE group, always. Upstream vLLM flags groups only when the EAGLE block drop is on, and our 0.30 version of patch 08 never set the flag; diffing the two versions showed it was the only functional difference. With `disable_eagle_block_drop: true` (in production since 2026-09-25), the flag left the cache half in EAGLE mode: the scheduler wrote the KDA recurrent state at the prompt's last full block, but the cache kept the state for the position one block lower, where nothing had been written. Without a recurrent state to resume from, the whole hit falls to zero. The two partial hits were prompts where that lower position happened to fall on the end of a prefill chunk, where a state did exist. The port was written and tested before production turned the block drop off, so its test never covered the combination we serve.

**How we proved it.** On CPU first, with no GPU and no server: we drove vLLM's real scheduler and KV cache manager from the main source tree with production's KV group layout and the six prompt lengths. With the flag, the replay reproduced the cluster's numbers exactly (0, 0, 0, 13,824, 13,824, 0); without it, production's. On hardware, the fixed build then matched production exactly, and so did the final stack with Humming (the cache resume line above).

**Fix.** Set the flag only when `use_eagle_block_drop()` is true, the same gate upstream uses (`image-main/patches/vllm/08-glm5-drafter-kv-group-flag.patch`).

**Lesson.** When a rebase regresses, diff the ported patches against the versions that worked before attributing the regression to upstream, and test the flag combination production actually runs.

### Short-turn caching (P1): built, tested, not shipped

Section 4 explains why short agent turns get no prefix-cache hits: the KDA state is saved only at 2,304-token block boundaries. vLLM main has `--prefix-match-unit`, which allows hits at a finer unit, but on our stack it did nothing. The DFlash2 drafter's sliding-window KV group has no fine-grained lookup, and vLLM turns fine-grained hits off for the whole model when any group lacks one. P1 adds that lookup for the drafter group, on top of two upstream pull requests ([vllm#58368](https://github.com/vllm-project/vllm/pull/58368) by netanel-haber and [vllm#58434](https://github.com/vllm-project/vllm/pull/58434) by njhill). We tested it on vLLM main (with Marlin) with a 256-token unit, against production (then the v0.30 build): short turns measured in the same window, long sessions against that window's restore boot and the previous window's last production run.

What it did:
- **Resumes land exactly at the prompt's last 256-token boundary:** a 1,800-token prompt resumes at 1,792, 3,000 at 2,816, 19,000 at 18,944.
- **Short agent turns** (prompts of about 1.5K to 2.5K tokens): prefix-cache hits 0% to 59%, time to first token 0.835x, throughput 1.07x.
- **Long agent sessions:** 1.04x tok/s, 1.01x time to first token, hit rate 0.858 vs 0.827. All six resumed long sessions resumed past their last full block: 18,688, 19,200, 18,688, 18,176, 17,920 and 18,944 cached tokens (production: 18,432, 18,432, 18,432, 16,128, 16,128, 18,432).
- **Quality:** NLL/token 1.26564 on the 40 chunks (production 1.26625 to 1.26724 over four boots), GSM8K 98.4%, long documents 1.57903 and 1.11577 (production 1.57878 and 1.11404), uncensored check passed, and no recurrent state written during decoding was reused (0 of 3 follow-ups).

**Why it is not shipped.** It failed one of the correctness checks we wrote before the run. A resumed prompt should continue as if it had been computed from scratch. On this stack even ordinary resumes are not bit-exact (greedy output is not bit-reproducible either; see the quality note in section 2), so the check compares against a control: how far the first generated token's logprobs move from a fresh computation, for resumes in the middle of a block vs ordinary full-block resumes on the same server. The bar was 2x the control. Mid-block resumes drifted by up to 0.237, against 0.089 for full-block resumes. That maximum comes from one of the eight mid-block resumes (19,000 tokens, second seed); the other seven stayed at or below 0.055, and the mean drift was the same for both kinds (0.086 and 0.089). Resending an identical, fully cached request moved the first token by up to 0.086, which is this stack's noise floor. So the failure rests on one outlier, and more seeds would tell a real bug from noise. Every other check passed and the speed gain is real, but a cache change that moves outputs more than twice as much as the normal path does not ship until we know why. P1 stays out until that drift is understood.

### Chat template and the thinking default

Until 2026-09-26 production served the multimodal template from Tony's recipe, which is older than Z.ai's official 09-07 template, with a server default of thinking off. Our agent clients always ask for reasoning, and vLLM turns thinking on when a request sends `reasoning_effort`, so agent requests reasoned anyway. Production now serves Z.ai's official 09-07 template plus one explicit opt-out line (`template/chat_template_zai0907_optout.jinja`; the launcher expects it in the checkpoint directory). There is no server-side thinking-off default any more: every request reasons unless it sends `enable_thinking: false`. With thinking on, the rendered prompt is byte-identical to the official template's. A request that names no reasoning effort gets the template's default, Max.

For the same reason, the 2026-09-26 serving loads send `reasoning_effort: max` on every request (GSM8K still runs with thinking off). The one-shot numbers in sections 1 and 2 were taken with thinking off.

### Drafters we could not measure, and smaller tests

- **RedHat's DSpark preview drafter.** vLLM 0.30 supports DSpark drafters, but on our hybrid stack this drafter's attention layers made vLLM pick 4,608-token cache blocks and pad the recurrent-state pages 99%. The boot check we wrote before the test stops a boot like that, so it was never measured. It needs a block-size fix; RedHat's final release may land first.
- **canada-quant DFlash2-G.** Not tested: its serving patches are not public, it needs new code for its full-attention layers, and its own model card shows lower throughput than incoai's drafter.
- **Always drafting 7 tokens** (no per-batch-size schedule): 0.97x on the long-session load, against a single production run. That is inside the noise described next, so there is no evidence it beats the schedule, and the schedule stays.
- **[vllm#58450](https://github.com/vllm-project/vllm/pull/58450)** (a faster GLM-5.3 metadata op): nothing to take, our backend already had the fast path.

### Benchmarking: production against itself

On the long agent-session load, production measured against itself moved almost as much as the changes we were judging. Later boots of the unchanged v0.30 build read 0.946x and 1.003x the first boot (tok/s, geometric mean over 1, 2, 4 and 8 sessions), and one level read 0.835x (4 sessions). The P1 window's restore boot read 0.970x the previous window's last production run. A single run against a single run cannot resolve a difference of about 5% on this load.

What we did about it in these windows: production was pooled over 3 boots in the vLLM main comparison and over 2 runs in the P1 window, each arm ran 2 sweeps, and every window ran production against itself so each result could be read against that null. Quality bars need the same treatment: NLL/token on the 40 chunks spans 1.26625 to 1.26686 across three boots of one build. A comparison against one production run, like the always-7 test above, is reported as no evidence either way.

## 1. Where decode time goes

vLLM's torch profiler, 30 steady decode steps per capture (the first 10 steps after `/start_profile` skipped so prefill stays out), stack tracing off:

```
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/cache/prof","torch_profiler_with_stack":false,"ignore_frontend":true,"delay_iterations":10,"max_iterations":30}'
curl -X POST http://head:8000/start_profile ; <send the load> ; curl -X POST http://head:8000/stop_profile
python3 tools/analyze_trace.py <rank0 trace.json.gz> 30
```

Rank 0 and rank 1 agree within about 1 ms per step.

| Share of GPU kernel time | 1 user (8-token verify) | 32 users (256-token verify) |
|---|---|---|
| step time (under the profiler) | 58.9 ms | 271 ms |
| MoE experts (Marlin NVFP4) | 45% (25.9 ms) | 47% (128 ms) |
| BF16 GEMMs (drafter, unquantized KDA gate projections, heads) | 19% (11.0 ms) | 7% (18.9 ms) |
| NCCL (includes ranks waiting on each other) | 13% (7 to 8 ms) | 12% (32.7 ms) |
| NVFP4 dense linears (Marlin) | 11% (6.4 ms) | 7% (18.1 ms) |
| KDA linear-attention kernel | 2% (1.0 ms) | 19% (50.8 ms) |
| sparse MLA attention + indexer | 1% (0.4 ms) | 1% (2.0 ms) |
| GPU idle | 6% | 1% |

What this says:
- **The MoE is at the memory-bandwidth limit.** At 32 users every one of the 288 experts is read every step, so its cost per step is fixed. Only more accepted tokens per step, or smaller weights, make it cheaper.
- **The CPU is not the bottleneck.** The GPU is idle 1 to 6% of the time.
- **The KDA kernel at high load is expensive for a structural reason.** For speculative decoding it saves a full fp32 copy of each sequence's recurrent state after every drafted token (8 copies per step) so rejected drafts can be rolled back. At 32 users that is about 9 GB of writes per step. A standalone benchmark (`tools/lab/src/kda_bench.py`) shows the kernel running at about 213 GB/s, close to the practical bandwidth limit, and no tile or warp setting changes it (1,411 to 1,448 us per call at 32 sequences across 8 launch configs).

Speculative decoding: median acceptance 0.391 per drafted token (12 runs, range 0.382 to 0.404), or 3.74 tokens per step with 7 drafted. By draft position: 0.77, 0.57, 0.42, 0.33, 0.26, 0.21, 0.17.

## 2. Experiments, including the ones that did not pay off

| What | Isolated result | Serving result | Kept? |
|---|---|---|---|
| RoCEnante one-shot RoCE all-reduce ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), adapter from local-inference-lab/vllm PR #597) | 4.4x faster than NCCL at 8 KB, 2.7x at 64 KB, 1.7x at 256 KB, 1.25x at 512 KB, slower at 1 MB | C1 +3% (noise), C8 +2.5%, C32 -5.7% | No |
| bf16 KDA recurrent state (`--mamba-ssm-cache-dtype bfloat16`, needs patch 09) | KDA kernel 2x faster (722 vs 1,418 us at 32 sequences) | C8 +1.7%, C32 +1.7%, quality unchanged; agent sessions at 1 to 8 with 1,280-token blocks: a tie (-0.6%, 95% interval -5% to +4%) and still zero cache hits | No |
| incoai DFlash2 revisions dc77ff1 (Aug 28) and bf582e4 (Aug 31) instead of 7d74cdd (Aug 27): same architecture, new weights | | One-shot harness: acceptance unchanged (0.393 over 12 dc77ff1 runs and 18 of 7d74cdd); dc77ff1 8 users +3.6% (t 3.0) but 1-user JSON -5% (t -5.7); bf582e4 within noise everywhere (3 runs). Agent sessions at 1, 2, 4, 8: dc77ff1 -2.9%, +3.2%, -10.3%, -2.8%, geometric mean -3% (paired 95% interval -7% to +1%) | No |
| modal-labs/GLM-5.3-Flash-DFlash drafter instead of incoai DFlash2 | | acceptance 0.392 vs 0.384 to 0.399 (a tie) | No |
| Stock RedHatAI NVFP4 checkpoint instead of ours (same incoai drafter) | | acceptance 0.407 vs our 0.384 to 0.399, but one user is 10 to 25% slower per prompt (45% on the long-prompt summary); 8 users 6% slower, 32 users even (one run) | No |
| NCCL knobs (graph mixing, graph helper, launch mode, protocols, CUDA sched flags) | none removed NCCL's fixed per-graph-launch cost | | No |
| B12X MoE backend (2026-09-22) | | slower at every concurrency | No |
| CuTeDSL linear backend (2026-09-22) | | +4.8% at C32, but 2 GB less memory headroom | No |
| 8-bit drafter projections (patch 10, 2026-09-24) | drafter weight reads 566 to 288 MB per rank | agent sessions at 1, 2, 4, 8: +8.6% (paired 95% interval +5.0% to +12.6%), 8 sessions 84.5 vs 71.0 tok/s, acceptance unchanged | **Yes** |
| 8-bit lm_head (patch 11, 2026-09-24) | lm_head 317 to 161 MB per rank | NLL/token +0.59% and +1.10% on two chunks (bar 0.1%; the same run with a BF16 lm_head moved 0.17% and -0.26%) | No |
| Our checkpoint: NVFP4 requantized from Blackfrost's BF16 with an MSE scale search (2026-09-24) | weight error 6.4% lower on 47 sampled tensors | NLL/token 1.26587 vs 1.29080, GSM8K 99.2% vs 98.4%; same-session speed: 1 user 46.9 vs 45.6 tok/s (9 runs each), agent sessions 1.02x (95% interval 0.96 to 1.08), the same agent tasks in 0.80x the time | **Yes** |
| `disable_eagle_block_drop` (2026-09-25) | | long agent sessions (about 19K-token prompts) at 1, 2, 4, 8: time to first token 0.68x, prefix-cache hit rate 70% to 83%; a resumed session now reuses its last full block (6 of 6 vs 0 of 6) | **Yes** |
| The model's own MTP head (layer 45, routed experts left in BF16, 3 drafted tokens) instead of DFlash2 (2026-09-25) | | long agent sessions: 0.96x tok/s, 1.05x time to first token; accepted tokens per step 2.67, 2.72, 2.54, 2.65 vs DFlash2's 3.09, 2.81, 2.64, 2.51 at 1, 2, 4, 8 sessions | No |
| Sampled drafting (`draft_sample_method: probabilistic`, 2026-09-25) | | acceptance +8.1% (0.391 vs 0.362) but 1.04x tok/s (the bar was 1.05) and 1.01x time to first token | No |
| Rebasing onto vLLM main (2026-09-24 snapshot) with our patches and the long-context fixes (2026-09-25) | | resumed agent sessions lost prefix-cache reuse (0 of 6 full-block resumes vs 6 of 6); NLL +0.17% | No at the time: 0.30 plus backports. The cache loss was a bug in our own port of patch 08, fixed 2026-09-26 (see the 2026-09-26 section) |
| Three long-context fixes (patches 12 to 14, 2026-09-25) | kernel tests: pass on this image, fail on the unfixed one | long-document NLL/token 1.578 and 1.118 vs 1.648 and 1.225; long agent sessions 1.04x tok/s, 0.99x time to first token | **Yes** |
| Fix for our port of patch 08 to vLLM main: flag the drafter's KV group as EAGLE only when the block drop is on (2026-09-26) | CPU replay of vLLM's scheduler and KV cache manager: 0, 0, 0, 13,824, 13,824, 0 cached tokens with the old flag (the cluster's exact numbers), production's numbers with the fix | six resumed long sessions: 18,432, 18,432, 18,432, 16,128, 16,128, 18,432 cached tokens, identical to production | **Yes** |
| vLLM main nightly 7f1a with our 16-patch series (2026-09-26) | | long agent sessions at 1, 2, 4, 8 (reasoning effort max, 3 production boots pooled): 0.955x tok/s, 1.002x time to first token, hit rate 0.827 on both; NLL/token 1.26750 vs 1.26625 to 1.26686; GSM8K 98.4% on both; uncensored check passed | **Yes**, as the base of production |
| Humming MoE backend (`--moe-backend humming`) on vLLM main (2026-09-26) | | against vLLM main without it: 0.977x tok/s (0.93, 1.18, 0.88, 0.95 at 1, 2, 4, 8), 1.018x time to first token. Against v0.30: 0.933x tok/s, 1.019x time to first token. NLL/token 1.26630, GSM8K 98.4%, uncensored check passed | **Yes** (production; the cost is stated in the 2026-09-26 section) |
| Short-turn caching, P1, on vLLM main (`--prefix-match-unit 256`, a drafter sub-block lookup, vllm#58368 and #58434; 2026-09-26) | resumes land at the prompt's last 256-token boundary | short agent turns: hits 0% to 59%, time to first token 0.835x, 1.07x tok/s; long sessions 1.04x tok/s, 1.01x time to first token; quality unchanged. First-token logprob drift after a mid-block resume: up to 0.237 vs 0.089 for full-block resumes (bar: 2x) | No: failed the drift check |
| Always 7 drafted tokens instead of the per-batch-size schedule | | long agent sessions: 0.97x against one production run, inside run-to-run noise | No |
| RedHat's DSpark preview drafter | vLLM picked 4,608-token cache blocks and padded the recurrent-state pages 99%; the boot check stopped it | not measured | No: needs a block-size fix |

Notes:
- **Component speedups did not carry into serving.** Twice, a component that was 2 to 4x faster in isolation moved end-to-end throughput by about 2%. Judge changes by a serving A/B, not by a microbenchmark or a profile share.
- **RoCEnante integration traps on stock vLLM 0.30.** b12x main registers vLLM plugins that import functions only local-inference-lab's vLLM fork has, which crashes argument parsing (fix: `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver`). b12x main also changed `all_reduce()` to require a `plan` argument without bumping its API version (fix: pin b12x to `00280f2`, the revision qualified with the adapter).
- **NCCL inside CUDA graphs** pays a fixed 0.55 to 1.0 ms per graph launch in a microbenchmark (one all-reduce per graph); the marginal cost per extra all-reduce is about the eager cost (43 us at 8 KB, 56 us at 64 KB). In serving this mattered much less than the microbenchmark suggested.
- **Quality checks** (`tools/quality_eval.py`): 30 generated multi-step math problems (answers computed in code), 4 facts hidden at 10 to 95% depth of a 34K-token document, and 5 long generations. Production scores 30/30 and 4/4. Greedy output is not bit-reproducible run to run on this stack (outputs of two identical runs diverge within the first few hundred characters), so long outputs are compared by reading, not by text match.

## 3. Long-context correctness (2026-09-25)

Our quality eval (`tools/quality_eval.py` plus NLL on 40 WikiText-2 chunks) passed every build. All of its chunks are under 2,048 tokens. GLM-5.3-Flash's sparse attention compresses keys into pools of 4 tokens, and three bugs in stock vLLM 0.30 only touch that path past 2,048 tokens of context ([vllm#58704](https://github.com/vllm-project/vllm/pull/58704), [#58454](https://github.com/vllm-project/vllm/pull/58454), [#57477](https://github.com/vllm-project/vllm/pull/57477); patches 12 to 14). A probe on two WikiText-2 documents of about 11K tokens each (NLL per token from prompt logprobs) sees them: 1.648 and 1.225 before the fixes, 1.578 and 1.118 after. Across four boots of the unfixed build the same probe read 1.641 to 1.656 and 1.222 to 1.225, so the drop is far outside boot-to-boot noise.

One honest wrinkle: on the short 40-chunk eval the fixed build read 0.15% higher NLL than the unfixed one, above the 0.1% bar we had set. That bar came from repeat runs within one boot; two boots of the same build differ by up to 0.07%, and the chunk-level spread doubled in both builds that carry #58704, so on our GB10 path that fix also shifts short-context numerics slightly. We shipped the fixes on the long-document result and are replacing the short-eval bar with one measured across boots.

## 4. Operational lessons

- **Size the KV cache against earlyoom, not against the GPU.** Each Spark runs earlyoom, which sends SIGTERM to the largest process at 4% available memory (about 4.9 GB). A 44 GiB KV pin left the head node about 1.4 GB above that line, and a long prefill and a docker image export on a serving node each killed production. 38 GiB left the head node 0.6 to 2.7 GB above it, and earlyoom killed rank 0 once. We serve 36 GiB. Bulk downloads onto a serving node go through direct I/O (`curl | dd oflag=direct`): plain writes fill the page cache and cost about 1 GB of available memory while they run.
- **Agent turns get no prefix-cache hits on this hybrid model.** vLLM can only resume the KDA layers from a state saved at a block boundary (mamba cache mode `align`). On 1 to 8 concurrent agent sessions (`tools/agent_sessions.py`) we measured zero hits on 439K prompt tokens with 2,304-token blocks (the launcher's value, and the smallest an fp32 state allows), and still zero on 438K with 1,280-token blocks (allowed by a bf16 state with patch 09; vLLM confirmed the 1,280 blocks at 1.27% page padding). In agent traffic, the block boundary a new turn could resume from falls inside the previous turn's generated text, and no state was reused there. Long shared prefixes such as a document still hit one full block at a time, which is where the faster follow-up turns come from. Long agent sessions (about 19K-token prompts) do hit: 70% of prompt tokens in our long-session load. By default vLLM's EAGLE-style drafter path drops the last cached block on every hit and recomputes it; `disable_eagle_block_drop: true` keeps it (this can only cost some draft acceptance; the target model still verifies every token), which raised the hit rate to 83% and cut time to first token by a third. We checked first that 0.30 never reuses KDA state written during decoding (vllm#53912 reports corruption from that with this setting): 0 of 5 follow-ups did. Saving the state at the end of each request is still the upstream fix for short turns. On vLLM main, short-turn caching (P1) took short turns from 0% to 59% hits but failed a correctness check and is not shipped; see the 2026-09-26 section.
- **Cap the multimodal processor cache** (`--mm-processor-cache-gb 1`). The 4 GiB default lives in both the API server and the engine core on the head node.
- **/health is not serving.** After a network event, /health kept returning 200 while multi-node requests could not complete. Check with a real completion.
- **`docker run` pulls missing images silently.** On a serving node that is a surprise 20 GB download. Use `--pull never`.
- **Use a fresh JIT cache directory per image build**, and never reuse one across vLLM builds.
- **Don't edit a bash script while it runs.** Bash reads scripts by offset, so an edit mid-run can execute half a line. Wrap long-running scripts in one `{ ... }` block so bash parses everything first.

## 5. The network binding trap (NetworkManager + netplan on a Spark cluster)

Each Spark's QSFP port shows up as two PCIe functions (`enp1s0f0np0`, `enP2p1s0f0np0`) with one address each. If the netplan entry for the first one is written with `match: {}`, NetworkManager is free to put that profile on either device. Ours had carried that line for weeks. Then every link on the cluster dropped for about 50 seconds, the management port and both QSFP functions on every node at the same instant, most likely the network gear rather than the nodes. When the links came back, three of four nodes brought the rail address up on the wrong function. On top of that, the RoCE GID table reshuffled on every node, so `NCCL_IB_GID_INDEX=3` no longer pointed at the IPv4 RoCEv2 entry. Serving silently broke while /health stayed green.

`nmcli connection modify` does not fix it durably: NetworkManager writes the change to a new netplan file that repeats `match: {}`. What worked:

```bash
sudo sed -i '/^      match: {}$/d' /etc/netplan/<your qsfp file>.yaml /etc/netplan/90-NM-<uuid of the rail-A profile>.yaml
sudo netplan generate && sudo nmcli connection reload
nmcli -g connection.interface-name connection show <rail-A profile>   # must print the interface name
```

Then check `ip -br -4 addr` on both functions and `/sys/class/infiniband/*/ports/1/gids/3` before launching.

## 6. Letting an AI agent experiment on a live cluster

Some of this work (the RoCEnante adapter port, groundwork for adaptive verification) was done by GLM-5.3-Flash itself, running on this cluster, as a coding agent. What made that safe and useful:
- **One narrow door to the hardware.** `tools/lab/lab_run.sh` is the only way experiments touch the nodes. It checks production with a real completion and checks every node's memory headroom before and after each run. It runs every rank under [oomwrap](https://github.com/osolmaz/oomwrap), with a floor well above the earlyoom line, and allows one run at a time.
- **An agent guard.** The agent's shell gets refusing `ssh`, `scp` and `docker` functions through the environment (`env 'BASH_FUNC_ssh%%=() { echo refused >&2; return 2; }' <agent command>`). Exported functions survive login shells, where a PATH shim does not (macOS `path_helper` puts `/usr/bin` first). `lab_run.sh` calls `/usr/bin/ssh` by absolute path.
- **Contracts, not instructions.** Each job gets a check script that must print PASS. Before handing it over, we confirm it fails on bad output and passes on good output. For code ports, the check applies the patch to a clean tree and runs the patched modules inside the real image.
- **Small jobs beat big ones.** A focused port job finished in about 15 minutes. A large "build the whole thing" job stalled for 25 minutes in a single generation and was stopped.
