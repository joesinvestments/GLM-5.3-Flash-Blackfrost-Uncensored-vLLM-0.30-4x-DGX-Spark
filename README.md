# GLM-5.3-Flash Blackfrost Uncensored on vLLM 0.30 (4x DGX Spark)

**A reproducible vLLM v0.30.0 release build for serving Blackfrost's uncensored GLM-5.3-Flash ([DERISKED NVFP4](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4), NVFP4 attention) across four NVIDIA DGX Spark (GB10) nodes at tensor parallel 4, with DFlash2 speculative decoding.**

Tony's build, which this one grew out of, runs on the launch-day vLLM image. This build ports the GB10 fixes onto the **v0.30.0 release**: stock `vllm/vllm-openai:v0.30.0` plus 10 small patches, a build script that verifies every patched file byte for byte, the exact serving flags, and the measurements behind them. Everything was measured on our own cluster, including the experiments that did not pay off.

## What the vLLM 0.30 build adds

Measured 2026-09-22 against the launch-day-image build, same checkpoint and serving setup:

| | Launch-day image | This build (v0.30.0) |
|---|---|---|
| Prompt processing | 1,761 to 2,016 tok/s | **2,211 to 2,496 tok/s** (about 25% faster) |
| Launch to serving | about 10 minutes | **about 4 minutes** (InstantTensor weight loading) |
| Follow-up on a long prompt, time to first token | baseline | **about 12x faster** (0.30 prefix-caches the hybrid KDA model) |

## Results (2026-09-23, 12 production runs)

| Load | Median | Range across runs |
|---|---|---|
| 1 user, simple output (counting) | 128 tok/s | 123 to 131 |
| 1 user, JSON | 92 tok/s | 87 to 93 |
| 1 user, code | 88 tok/s | 87 to 96 |
| 1 user, step-by-step math | 72 tok/s | 69 to 77 |
| 1 user, explanation | 50 tok/s | 47 to 53 |
| 1 user, creative prose | 38 tok/s | 35 to 39 |
| 8 users, aggregate | 287 tok/s | 269 to 313 |
| 32 users, aggregate | 591 tok/s | 555 to 602 |

- Speculative decoding: 0.391 of drafted tokens accepted (3.74 tokens per step, 7 drafted).
- Prompt processing: 2,211 to 2,496 tok/s (measured 2026-09-22).
- Launch to serving: about 4 minutes (weights load in about 3 with InstantTensor).
- Conditions: temperature 0, thinking off, max 400 tokens, prompts in `tools/ab_harness.py`. Between identical runs, throughput moves 2 to 4%.

**Agent sessions** (`tools/agent_sessions.py`: 3-turn tool-calling loops, half the sessions with thinking on, default sampling, 3 to 4 sweeps per level):

| Concurrent sessions | Aggregate (mean) | Range across sweeps |
|---|---|---|
| 1 | 44 tok/s | 41 to 46 |
| 2 | 52 tok/s | 49 to 54 |
| 4 | 69 tok/s | 65 to 71 |
| 8 | 83 tok/s | 77 to 91 |

These sit well below the one-shot numbers: agent turns are short (100 to 160 tokens on average), default sampling accepts fewer drafted tokens (0.30 to 0.34 against 0.39 at temperature 0), and every turn's prompt is processed in full, because agent turns get no prefix-cache hits on this model (see `docs/findings.md`).

## What's here

```
image/     build.sh, patches (8 vLLM + 2 FlashInfer + 1 optional), EXPECTED.sha256
launch/    launch_node.sh: the exact serving flags, one rank per node
tools/     ab_harness.py (speed + acceptance), agent_sessions.py (1 to 8 concurrent agent sessions),
           quality_eval.py, analyze_trace.py (profiler breakdown)
tools/lab/ lab_run.sh: a safe way to run experiments (or an AI agent) on a cluster that is also serving
docs/      patches.md (what each patch does and where it came from), findings.md (profile, experiments, lessons)
```

## Quick start

**1. Build the image on each node.** You need `vllm/vllm-openai:v0.30.0` locally (about 20 GB).
```bash
cd image && ./build.sh
```
The script patches the files out of the stock image and refuses to build unless every result matches `EXPECTED.sha256`, the checksums of the image behind the numbers above. The build only copies files; no container runs.

**2. Get the weights onto every node** (not redistributed here; follow each license):
- Checkpoint: [Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/GLM-5.3-Flash-DERISKED-NVFP4), converted to NVFP4 attention with Tony's recipe in [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) (`runs/2026-09-21-blackfrost-derisked/`). Put it in `$HF_ROOT/hub/glm53-flash-derisked-nvfp4-attn/`. It must include `chat_template_mm.jinja`.
- Drafter: [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at revision `7d74cdd881ed7e32c31175984a67823127b66cfe`, in `$HF_ROOT/hub/glm53-flash-dflash2/`. We measured all three published revisions on this build, on one-shot prompts and on 1 to 8 concurrent agent sessions; none beat 7d74cdd, which is behind every number here.

**3. Launch, workers first, head last:**
```bash
export NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4"   # rail IPs, rank 0 (head) first
./launch/launch_node.sh 3   # on node 3, then 2 and 1
./launch/launch_node.sh 0   # on the head node
```
Defaults assume the Spark's QSFP port shows up as `enp1s0f0np0` / `enP2p1s0f0np0` (RDMA devices `rocep1s0f0` / `roceP2p1s0f0`) and RoCE GID index 3; see the variables at the top of the script. We run every rank under [oomwrap](https://github.com/osolmaz/oomwrap) with `tools/lab/oomwrap_supervise.sh`.

**4. Verify with a real completion,** not just `/health`:
```bash
curl -s http://HEAD:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","max_tokens":16,"messages":[{"role":"user","content":"Say OK"}]}'
```

**Memory:** each Spark runs earlyoom (SIGTERM at 4% free, about 4.9 GB). The 38 GiB KV cache in `launch_node.sh` leaves the head node about 7 GB above that line; larger values got the head node's vLLM worker killed. See `docs/findings.md`.

## Key findings

Details and numbers are in [docs/findings.md](docs/findings.md).

- **Where decode time goes.** At 1 user: MoE experts 45%, BF16 GEMMs 19%, NCCL 13%, GPU idle 6%. At 32 users: MoE 47%, the KDA linear-attention kernel 19%, NCCL 12%, idle 1%. The MoE is at the memory-bandwidth limit.
- **Component speedups did not carry into serving.** A graph-capturable RoCE all-reduce 2 to 4x faster than NCCL, and a 2x faster KDA kernel from bf16 state, each moved end-to-end throughput by only about 2%. Judge changes by a serving A/B.
- **No public drafter beats the one we run.** incoai's newer revisions win some prompts and lose others (dc77ff1: 3.6% faster at 8 users on a code prompt, 5% slower on single-user JSON), and across 1 to 8 concurrent agent sessions dc77ff1 came out 3% behind. The modal-labs DFlash drafter ties. Against stock RedHatAI NVFP4, our converted checkpoint (DERISKED weights, NVFP4 attention) gives up almost nothing in acceptance, and stock is 10 to 25% slower for one user (45% on a long prompt), about even at 32 users.
- **Agent turns never hit the prefix cache.** Zero hits on about 440K prompt tokens of agent traffic with 2,304-token blocks, and again with 1,280-token blocks. A new turn needs the KDA state from the end of the previous request, which vLLM does not save; that is the upstream fix.
- **A vLLM bug:** `--mamba-ssm-cache-dtype` is silently ignored for GLM-5.3-Flash. Fix in `image/patches/vllm/optional/09-*.patch`.
- **A network trap on Spark clusters:** a netplan `match: {}` let NetworkManager bring the rail address up on the wrong QSFP function after a network blip, while `/health` stayed green.

## Credits

- **Tony ([@2WildTech](https://x.com/2WildTech), [tonyd2wild](https://github.com/tonyd2wild))**: the GB10 patch set and image that patches 01 to 08 are ported from, the two FlashInfer FP8 MLA fixes, and the NVFP4-attention conversion recipe.
- **[Blackfrost](https://huggingface.co/Blackfrost-AI)**: GLM-5.3-Flash-DERISKED-NVFP4 (MIT). The checkpoint served here adds NVFP4 attention on top of their weights; per their model card, Blackfrost has not evaluated modified versions. **[Z.ai](https://huggingface.co/zai-org)**: GLM-5.3-Flash (MIT).
- **[incoai](https://huggingface.co/incoai)**: the GLM-5.3-Flash DFlash2 drafter (CC BY-NC-ND 4.0).
- **[vLLM](https://github.com/vllm-project/vllm)** and **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** (Apache-2.0).
- **[local-inference-lab](https://github.com/local-inference-lab)**: b12x and RoCEnante, tested here. **[modal-labs](https://huggingface.co/modal-labs)**: the DFlash drafter we compared against.
- **[osolmaz/oomwrap](https://github.com/osolmaz/oomwrap)**: memory-pressure supervision for every launch.

## License

The scripts, tools and patches in this repo are Apache-2.0 (see `LICENSE`); the patches modify Apache-2.0 vLLM and FlashInfer code. No model weights are included. Each model has its own license: the incoai drafter is non-commercial and no-derivatives.
