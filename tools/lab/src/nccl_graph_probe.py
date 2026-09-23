# Fixed cost of launching a CUDA graph that holds ONE NCCL all_reduce, per replay, back-to-back and with a
# 1 ms host gap between replays (production does CPU work between decode steps).
# env: SCHED=auto|spin|yield|block (primary-context scheduling flag, set before CUDA init), SIZE_KB, N, TAG
# Prints from rank 0:  GPROBE tag=<TAG> gap=<0|1ms> p50_us=<x> p10_us=<x> p90_us=<x>
import ctypes, os, statistics, time
SCHED = {"auto": 0, "spin": 1, "yield": 2, "block": 4}[os.environ.get("SCHED", "auto")]
cu = ctypes.CDLL("libcuda.so.1"); dev = ctypes.c_int()
assert cu.cuInit(0) == 0 and cu.cuDeviceGet(ctypes.byref(dev), 0) == 0
assert cu.cuDevicePrimaryCtxSetFlags(dev, SCHED) == 0
import torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"]); N = int(os.environ.get("N", "1000"))
TAG = os.environ.get("TAG", "base")
torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=rank, world_size=W, device_id=torch.device("cuda", 0))
x = torch.zeros(int(os.environ.get("SIZE_KB", "64")) * 512, device="cuda", dtype=torch.bfloat16)
for _ in range(20): dist.all_reduce(x)
g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
with torch.cuda.stream(st):
    with torch.cuda.graph(g, stream=st):
        dist.all_reduce(x)
torch.cuda.synchronize(); dist.barrier()
for gap in (0.0, 0.001):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(N)]
    for _ in range(50): g.replay()
    torch.cuda.synchronize(); dist.barrier()
    for a, b in ev:
        a.record(); g.replay(); b.record()
        if gap: torch.cuda.synchronize(); time.sleep(gap)
    torch.cuda.synchronize()
    t = sorted(a.elapsed_time(b) * 1e3 for a, b in ev)
    if rank == 0:
        print(f"GPROBE tag={TAG} gap={'1ms' if gap else '0'} p50_us={statistics.median(t):.1f} "
              f"p10_us={t[len(t) // 10]:.1f} p90_us={t[9 * len(t) // 10]:.1f}", flush=True)
    dist.barrier()
torch.cuda.synchronize()
os._exit(0)
