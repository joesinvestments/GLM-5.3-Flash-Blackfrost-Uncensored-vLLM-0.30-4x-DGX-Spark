# Decode-step breakdown from one vLLM torch-profiler trace (Chrome JSON, .gz ok).
# Usage: python3 analyze_trace.py <trace.json.gz> [steps]
# Reports: captured wall time, GPU busy (union of all GPU intervals) vs idle, time by kernel category, top kernels,
# graph launches and step annotations (to count steps).
import collections, gzip, json, re, sys

path = sys.argv[1]; steps_hint = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not gpu: sys.exit("no GPU events in trace")
t0 = min(e["ts"] for e in gpu); t1 = max(e["ts"] + e["dur"] for e in gpu); wall = t1 - t0
busy = 0; cur_s = cur_e = None                                   # union of intervals across all streams
for s, e in sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu):
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s

CATS = [  # first match wins; tuned to GLM-5.3-Flash on vLLM 0.30 (Marlin NVFP4, FlashInfer sparse MLA, KDA, DFlash2)
    ("comm (NCCL)", r"nccl"),
    ("MoE experts (Marlin)", r"moe.*marlin|marlin.*moe|fused_moe|moe_wna16"),
    ("quantized linears (Marlin)", r"marlin"),
    ("dense BF16 GEMM (drafter, lm_head)", r"gemm|cutlass|cublas|nvjet|xmma|sm\d+_|gemv"),
    ("sparse-MLA attention + indexer", r"mla|flashinfer|batchprefill|batchdecode|indexer|mqa_logits|sparse|paged"),
    ("KDA linear attention + conv", r"fla|chunk|recurrent|kda|gdn|gated_delta|conv1d|causal_conv|l2norm"),
    ("MoE routing / top-k", r"topk|moe_align|sort|gating|router|grouped_topk|count_and_sort"),
    ("norms / activations / elementwise", r"rms|norm|silu|swiglu|act_and|mul|add|elementwise|vectorized|unrolled|reduce|fill|copy_kernel|cat"),
    ("sampling / spec-decode bookkeeping", r"sampl|argmax|reject|gather|scatter|index|embedding|prepare|cumsum|arange"),
]
cat_t = collections.Counter(); name_t = collections.Counter(); name_n = collections.Counter()
for e in gpu:
    n = e["name"]; d = e["dur"]; name_t[n] += d; name_n[n] += 1
    if e["cat"] != "kernel": cat_t["memcpy / memset"] += d; continue
    for c, rx in CATS:
        if re.search(rx, n, re.I): cat_t[c] += d; break
    else: cat_t["other kernels"] += d

cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
launches = sum(1 for e in cpu if "GraphLaunch" in e.get("name", ""))
ann = collections.Counter(e["name"] for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation")
steps = steps_hint or None
print(f"trace: {path}")
print(f"captured wall {wall / 1e3:.1f} ms | GPU busy {busy / 1e3:.1f} ms ({100 * busy / wall:.0f}%) | GPU idle {(wall - busy) / 1e3:.1f} ms ({100 * (wall - busy) / wall:.0f}%)")
print(f"cudaGraphLaunch calls: {launches} | GPU events: {len(gpu)}")
print("top user annotations:", ", ".join(f"{k} x{v}" for k, v in ann.most_common(6)))
if steps: print(f"per step (/{steps}): wall {wall / steps / 1e3:.2f} ms, busy {busy / steps / 1e3:.2f} ms, idle {(wall - busy) / steps / 1e3:.2f} ms")
tot = sum(cat_t.values())
print("\nGPU time by category (sum of kernel durations; streams can overlap):")
for c, t in cat_t.most_common():
    extra = f"  = {t / steps / 1e3:.2f} ms/step" if steps else ""
    print(f"  {c:40s} {t / 1e3:9.1f} ms  {100 * t / tot:5.1f}%{extra}")
print("\ntop 25 kernels:")
for n, t in name_t.most_common(25):
    print(f"  {t / 1e3:8.1f} ms  {100 * t / tot:5.1f}%  x{name_n[n]:<6} avg {t / name_n[n]:7.1f} us  {n[:110]}")
