# KDA spec-decode recurrent kernel microbenchmark at production shapes (GLM-5.3-Flash TP4: 16 local heads, head_dim 128,
# 8 verify tokens per sequence). Times GLM's own wrapper (fused_recurrent_kda_fwd) as shipped, with a bf16 recurrent
# state, and with other tile widths / warp counts (a copy of the wrapper with only BV and num_warps rewritten).
# Prints: KDA n=<seqs> variant=<name> us=<median per call> GBps=<state bytes / time> maxdiff=<vs shipped output>
import inspect, os, statistics, torch
import vllm.models.glm5next.nvidia.ops.third_party.kda.kernels as km
H, D, T = 16, 128, 8
src = inspect.getsource(km.fused_recurrent_kda_fwd)
assert src.count("min(next_power_of_2(V), 8)") == 1 and src.count("num_warps = 1") == 1

def variant(bv, nw):
    s = src.replace("min(next_power_of_2(V), 8)", f"min(next_power_of_2(V), {bv})").replace("num_warps = 1", f"num_warps = {nw}")
    s = s.replace("def fused_recurrent_kda_fwd(", f"def _fwd_{bv}_{nw}(")
    ns = dict(km.__dict__); exec(s, ns); return ns[f"_fwd_{bv}_{nw}"]

def inputs(n, state_dtype):
    torch.manual_seed(0)
    tok = n * T
    q, k, v = (torch.randn(1, tok, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    g = torch.randn(1, tok, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    beta = torch.randn(1, tok, H, device="cuda", dtype=torch.bfloat16)
    pool = (torch.randn(1 + tok, H, D, D, device="cuda", dtype=torch.float32) * 0.02).to(state_dtype)
    idx = (1 + torch.arange(tok, device="cuda", dtype=torch.int32)).view(n, T)
    acc = torch.full((n,), 4, device="cuda", dtype=torch.int32)
    cu = torch.arange(0, tok + 1, T, device="cuda", dtype=torch.int32)
    a_log = torch.randn(H, device="cuda", dtype=torch.float32) * 0.1
    g_bias = torch.randn(H * D, device="cuda", dtype=torch.float32) * 0.1
    return dict(q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5, initial_state=pool, inplace_final_state=True, cu_seqlens=cu,
                ssm_state_indices=idx, num_accepted_tokens=acc, use_qk_l2norm_in_kernel=True, sigmoid_beta=True,
                a_log=a_log, g_bias=g_bias, compute_gate=True, lower_bound=-5.0)

def run(fn, n, state_dtype, iters=50):
    a = inputs(n, state_dtype)
    pristine = a["initial_state"].clone()
    out = fn(**a)[0].float(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a["initial_state"].copy_(pristine)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(**a); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts), out

shipped = km.fused_recurrent_kda_fwd
VARIANTS = [("shipped(BV8,w1)", shipped, torch.float32), ("bf16-state", shipped, torch.bfloat16)]
VARIANTS += [(f"BV{bv},w{nw}", variant(bv, nw), torch.float32) for bv, nw in ((16, 1), (32, 1), (32, 2), (32, 4), (64, 4), (64, 8), (128, 8))]
for n in (1, 8, 32):
    ref = None
    for name, fn, sd in VARIANTS:
        try:
            us, out = run(fn, n, sd)
        except Exception as ex:  # a tile config the kernel cannot compile is a result, not a crash
            print(f"KDA n={n} variant={name} FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True); continue
        if ref is None: ref = out
        state_bytes = n * H * D * D * torch.tensor([], dtype=sd).element_size() * (T + 1)
        print(f"KDA n={n} variant={name} us={us:.1f} GBps={state_bytes / us / 1e3:.0f} maxdiff={(out - ref).abs().max().item():.2e}", flush=True)
os._exit(0)
