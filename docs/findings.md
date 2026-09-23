# Findings

Everything below was measured on 4x DGX Spark (GB10, 128 GB unified memory each) with ConnectX-7 RoCE through a switch, serving the config in `launch/launch_node.sh`. Serving numbers come from `tools/ab_harness.py`: temperature 0, thinking off, max 400 tokens, same prompts every run. Between identical runs of the same config, aggregate throughput moves 2 to 4% and acceptance about 0.02, so smaller differences are noise.

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

Notes:
- **Component speedups did not carry into serving.** Twice, a component that was 2 to 4x faster in isolation moved end-to-end throughput by about 2%. Judge changes by a serving A/B, not by a microbenchmark or a profile share.
- **RoCEnante integration traps on stock vLLM 0.30.** b12x main registers vLLM plugins that import functions only local-inference-lab's vLLM fork has, which crashes argument parsing (fix: `VLLM_PLUGINS=lora_filesystem_resolver,lora_hf_hub_resolver`). b12x main also changed `all_reduce()` to require a `plan` argument without bumping its API version (fix: pin b12x to `00280f2`, the revision qualified with the adapter).
- **NCCL inside CUDA graphs** pays a fixed 0.55 to 1.0 ms per graph launch in a microbenchmark (one all-reduce per graph); the marginal cost per extra all-reduce is about the eager cost (43 us at 8 KB, 56 us at 64 KB). In serving this mattered much less than the microbenchmark suggested.
- **Quality checks** (`tools/quality_eval.py`): 30 generated multi-step math problems (answers computed in code), 4 facts hidden at 10 to 95% depth of a 34K-token document, and 5 long generations. Production scores 30/30 and 4/4. Greedy output is not bit-reproducible run to run on this stack (outputs of two identical runs diverge within the first few hundred characters), so long outputs are compared by reading, not by text match.

## 3. Operational lessons

- **Size the KV cache against earlyoom, not against the GPU.** Each Spark runs earlyoom, which sends SIGTERM to the largest process at 4% available memory (about 4.9 GB). A 44 GiB KV pin left the head node about 1.4 GB above that line, and a long prefill and a docker image export on a serving node each killed production. 38 GiB leaves about 7 GB.
- **Agent turns get no prefix-cache hits on this hybrid model.** vLLM can only resume the KDA layers from a state saved at a block boundary (mamba cache mode `align`). On 1 to 8 concurrent agent sessions (`tools/agent_sessions.py`) we measured zero hits on 439K prompt tokens with 2,304-token blocks (the launcher's value, and the smallest an fp32 state allows), and still zero on 438K with 1,280-token blocks (allowed by a bf16 state with patch 09; vLLM confirmed the 1,280 blocks at 1.27% page padding). In agent traffic, the block boundary a new turn could resume from falls inside the previous turn's generated text, and no state was reused there. Long shared prefixes such as a document still hit one full block at a time, which is where the faster follow-up turns come from. Saving the state at the end of each request is the upstream fix.
- **Cap the multimodal processor cache** (`--mm-processor-cache-gb 1`). The 4 GiB default lives in both the API server and the engine core on the head node.
- **/health is not serving.** After a network event, /health kept returning 200 while multi-node requests could not complete. Check with a real completion.
- **`docker run` pulls missing images silently.** On a serving node that is a surprise 20 GB download. Use `--pull never`.
- **Use a fresh JIT cache directory per image build**, and never reuse one across vLLM builds.
- **Don't edit a bash script while it runs.** Bash reads scripts by offset, so an edit mid-run can execute half a line. Wrap long-running scripts in one `{ ... }` block so bash parses everything first.

## 4. The network binding trap (NetworkManager + netplan on a Spark cluster)

Each Spark's QSFP port shows up as two PCIe functions (`enp1s0f0np0`, `enP2p1s0f0np0`) with one address each. If the netplan entry for the first one is written with `match: {}`, NetworkManager is free to put that profile on either device. Ours had carried that line for weeks. Then every link on the cluster dropped for about 50 seconds, the management port and both QSFP functions on every node at the same instant, most likely the network gear rather than the nodes. When the links came back, three of four nodes brought the rail address up on the wrong function. On top of that, the RoCE GID table reshuffled on every node, so `NCCL_IB_GID_INDEX=3` no longer pointed at the IPv4 RoCEv2 entry. Serving silently broke while /health stayed green.

`nmcli connection modify` does not fix it durably: NetworkManager writes the change to a new netplan file that repeats `match: {}`. What worked:

```bash
sudo sed -i '/^      match: {}$/d' /etc/netplan/<your qsfp file>.yaml /etc/netplan/90-NM-<uuid of the rail-A profile>.yaml
sudo netplan generate && sudo nmcli connection reload
nmcli -g connection.interface-name connection show <rail-A profile>   # must print the interface name
```

Then check `ip -br -4 addr` on both functions and `/sys/class/infiniband/*/ports/1/gids/3` before launching.

## 5. Letting an AI agent experiment on a live cluster

Some of this work (the RoCEnante adapter port, groundwork for adaptive verification) was done by GLM-5.3-Flash itself, running on this cluster, as a coding agent. What made that safe and useful:
- **One narrow door to the hardware.** `tools/lab/lab_run.sh` is the only way experiments touch the nodes. It checks production with a real completion and checks every node's memory headroom before and after each run. It runs every rank under [oomwrap](https://github.com/osolmaz/oomwrap), with a floor well above the earlyoom line, and allows one run at a time.
- **An agent guard.** The agent's shell gets refusing `ssh`, `scp` and `docker` functions through the environment (`env 'BASH_FUNC_ssh%%=() { echo refused >&2; return 2; }' <agent command>`). Exported functions survive login shells, where a PATH shim does not (macOS `path_helper` puts `/usr/bin` first). `lab_run.sh` calls `/usr/bin/ssh` by absolute path.
- **Contracts, not instructions.** Each job gets a check script that must print PASS. Before handing it over, we confirm it fails on bad output and passes on good output. For code ports, the check applies the patch to a clean tree and runs the patched modules inside the real image.
- **Small jobs beat big ones.** A focused port job finished in about 15 minutes. A large "build the whole thing" job stalled for 25 minutes in a single generation and was stopped.
