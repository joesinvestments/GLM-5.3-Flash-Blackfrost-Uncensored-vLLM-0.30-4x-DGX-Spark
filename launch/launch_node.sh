#!/usr/bin/env bash
# Launch one rank of GLM-5.3-Flash across 4 DGX Sparks (TP=4, one GPU per node) with the image from ../image/build.sh.
# These are the exact serving flags behind the published numbers; only machine-specific values are variables.
# Run on every node, workers first (3, 2, 1) and the head (0) last:   NODES="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4" ./launch_node.sh <rank>
set -uo pipefail
NODE_RANK="${1:?usage: NODES=\"ip0 ip1 ip2 ip3\" launch_node.sh <0|1|2|3>}"
read -r -a NODES <<< "${NODES:?set NODES to the 4 rail IPs, rank 0 first}"
[ ${#NODES[@]} = 4 ] || { echo "NODES must list 4 IPs"; exit 2; }
IMAGE=${IMAGE:-glm53-flash-gb10:v0.30.0}
HF_ROOT=${HF_ROOT:-/var/tmp/hf}                       # host dir mounted at /cache/huggingface
MODEL=${MODEL:-glm53-flash-derisked-nvfp4-attn}       # dir under $HF_ROOT/hub (see README: checkpoint)
DRAFTER=${DRAFTER:-glm53-flash-dflash2}               # dir under $HF_ROOT/hub (incoai/GLM-5.3-Flash-DFlash2)
CACHE=${CACHE:-/var/tmp/glm53-vllm-cache}             # JIT caches; use a fresh dir per image build
RAIL_IFS=${RAIL_IFS:-enp1s0f0np0,enP2p1s0f0np0}       # both PCIe functions of the Spark's QSFP port
RAIL_HCAS=${RAIL_HCAS:-rocep1s0f0,roceP2p1s0f0}
GLOO_IF=${GLOO_IF:-enp1s0f0np0}
GID_INDEX=${GID_INDEX:-3}                             # check /sys/class/infiniband/*/ports/1/gids/3 after any network event
NAME=${NAME:-glm53_vllm}; PORT=${PORT:-8000}; MPORT=${MPORT:-29654}
GMU=0.85; MAXLEN=500000; SEQS=64; MNBT=8192; BLOCK=2304; MOE_BACKEND=marlin; SPEC_K=7
KV_MEM=40802189312; KV_DTYPE=fp8_e4m3; CG=FULL_AND_PIECEWISE   # 38 GiB KV: leaves the head node ~7 GB above a 4% earlyoom line
HOST_IP=${NODES[$NODE_RANK]}; HEAD_IP=${NODES[0]}
HEADLESS=""; [ "$NODE_RANK" != 0 ] && HEADLESS="--headless"

for f in "$HF_ROOT/hub/$MODEL/config.json" "$HF_ROOT/hub/$MODEL/chat_template_mm.jinja" "$HF_ROOT/hub/$DRAFTER/config.json"; do
  test -f "$f" || { echo "MISSING $f"; exit 3; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "MISSING image $IMAGE (build it with ../image/build.sh)"; exit 3; }
mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --gpus all -d --name "$NAME" --restart no --pull never \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
  -v "$HF_ROOT:/cache/huggingface" -v "$CACHE:/cache" \
  -e VLLM_HOST_IP="$HOST_IP" -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e MAX_JOBS=2 \
  -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$RAIL_HCAS" -e NCCL_SOCKET_IFNAME="$RAIL_IFS" \
  -e GLOO_SOCKET_IFNAME="$GLOO_IF" -e TP_SOCKET_IFNAME="$GLOO_IF" \
  -e NCCL_IB_GID_INDEX="$GID_INDEX" -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e TORCH_NCCL_ENABLE_MONITORING=0 -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400 \
  -e PYTHONUNBUFFERED=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  "$IMAGE" \
    "/cache/huggingface/hub/$MODEL" \
    --served-model-name glm-5.3-flash --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code --load-format instanttensor \
    --tensor-parallel-size 4 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --block-size "$BLOCK" --moe-backend "$MOE_BACKEND" --max-num-batched-tokens "$MNBT" \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"/cache/huggingface/hub/$DRAFTER\",\"num_speculative_tokens\":$SPEC_K}" \
    --kv-cache-dtype "$KV_DTYPE" --kv-cache-memory "$KV_MEM" \
    --compilation-config "{\"cudagraph_mode\":\"$CG\"}" \
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
    --chat-template "/cache/huggingface/hub/$MODEL/chat_template_mm.jinja" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    $HEADLESS --limit-mm-per-prompt '{"image":16,"video":0}' --mm-processor-cache-gb 1
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP image=$IMAGE"
