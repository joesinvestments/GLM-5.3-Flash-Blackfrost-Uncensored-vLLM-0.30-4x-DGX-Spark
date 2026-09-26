#!/usr/bin/env bash
# Optional: NCCL v2.29.7-1 (the version vllm/vllm-openai:v0.30.0 bundles) + NVIDIA/nccl PR #2393, an acquire fence in
# ncclIbIsend's CTS poll. On Arm (GB10) a weakly ordered load there can read a stale request count and hang a RoCE
# collective (NVIDIA/nccl#2334). Compiles inside the base image in a container capped at 3 GB of RAM and 2 CPUs (about
# 12 minutes on a Spark, safe next to a serving model), for sm_121 only, and writes ./libnccl.so.2. Then, from image/:
#   NCCL_LIB=nccl/libnccl.so.2 ./build.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); BASE=${BASE:-vllm/vllm-openai:v0.30.0}; W=$(mktemp -d)
git clone -q --depth 1 --branch v2.29.7-1 https://github.com/NVIDIA/nccl "$W/src"
(cd "$W/src" && git apply "$HERE/pr2393-v2.29.7.diff")
docker run --rm --pull never --memory 3g --memory-swap 3g --cpus 2 -v "$W:/work" -w /work/src --entrypoint bash "$BASE" -c \
  'nice -n 19 make -j2 src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" > /work/build.log 2>&1' \
  || { echo "build failed, see $W/build.log"; exit 1; }
cp -L "$W/src/build/lib/libnccl.so.2" "$HERE/libnccl.so.2"
echo "built $HERE/libnccl.so.2 ($(sha256sum "$HERE/libnccl.so.2" | cut -c1-16)); build tree left in $W"
