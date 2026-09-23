# NCCL all_reduce baseline at production decode sizes (hidden 4096 bf16 = 8 KB per token; verify step = 8 tokens/seq).
# Prints from rank 0:  NCCL_AR tag=<TAG> mode=<eager|graph20|graph1> bytes=<n> avg_us=<x>   (env: SIZES_KB, TAG)
import os, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); W = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=rank, world_size=W, device_id=torch.device("cuda", 0))
SIZES = [int(k) << 10 for k in os.environ.get("SIZES_KB", "8,64,256,512,1024").split(",")]  # draft C1, verify C1, C4, C8, C16
TAG = os.environ.get("TAG", "base")
if rank == 0: print(f"NCCL version {torch.cuda.nccl.version()} torch {torch.__version__}", flush=True)
for nbytes in SIZES:
    x = torch.zeros(nbytes // 2, device="cuda", dtype=torch.bfloat16)
    for _ in range(20): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(200): dist.all_reduce(x)
    e.record(); torch.cuda.synchronize()
    eager = s.elapsed_time(e) * 1e3 / 200
    res = {"eager": eager}
    for ops in (20, 1):  # graph20: 20 captured all-reduces per graph; graph1: one per graph, replayed 200x
        g = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            with torch.cuda.graph(g, stream=st):
                for _ in range(ops): dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        g.replay(); torch.cuda.synchronize(); dist.barrier()
        s.record()
        for _ in range(200 // ops): g.replay()
        e.record(); torch.cuda.synchronize()
        res[f"graph{ops}"] = s.elapsed_time(e) * 1e3 / 200
    if rank == 0:
        for m, v in res.items(): print(f"NCCL_AR tag={TAG} mode={m} bytes={nbytes} avg_us={v:.1f}", flush=True)
dist.barrier(); torch.cuda.synchronize()
os._exit(0)  # destroy_process_group with live NCCL graphs hangs on this stack; nothing left to clean up
