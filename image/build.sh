#!/usr/bin/env bash
# Build the GLM-5.3-Flash GB10 image: stock vllm/vllm-openai:v0.30.0 + 13 vLLM patches + 2 FlashInfer patches.
# Copies the target files out of the base image, patches them, checks every result against EXPECTED.sha256, then builds
# a COPY-only image (no container runs during the build).
#   ./build.sh                                -> glm53-flash-gb10:v0.30.0
#   TAG=myname:tag ./build.sh                 -> custom tag
#   NCCL_LIB=nccl/libnccl.so.2 ./build.sh     -> also replace the bundled NCCL with the fence-fixed build (nccl/build_nccl.sh)
#   WITH_PATCH_09=1 ./build.sh                -> also apply optional/09 (lets --mamba-ssm-cache-dtype reach the KDA state)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-vllm/vllm-openai:v0.30.0}
TAG=${TAG:-glm53-flash-gb10:v0.30.0}
SP=/usr/local/lib/python3.12/dist-packages
NCCL_DST=nvidia/nccl/lib/libnccl.so.2
PATCHES=("$HERE"/patches/vllm/[0-9]*.patch "$HERE"/patches/flashinfer/*.patch)
[ "${WITH_PATCH_09:-0}" = 1 ] && PATCHES+=("$HERE"/patches/vllm/optional/09-*.patch)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"; docker rm -f glm53-gb10-extract >/dev/null 2>&1 || true' EXIT
targets() { grep -h '^+++ b/' "$@" | sed 's#^+++ b/##' | cut -f1 | sort -u; }
FILES=$(targets "${PATCHES[@]}")

docker image inspect "$BASE" >/dev/null 2>&1 || { echo "base image $BASE not present; pull it first (about 20 GB)"; exit 2; }
docker create --name glm53-gb10-extract "$BASE" >/dev/null
for f in $FILES; do  # patches 10 and 11 add new files, which the base image does not have
  mkdir -p "$WORK/$(dirname "$f")"; docker cp "glm53-gb10-extract:$SP/$f" "$WORK/$f" 2>/dev/null || echo "new file: $f"
done
for p in "${PATCHES[@]}"; do (cd "$WORK" && patch -p1 --forward --no-backup-if-mismatch -s < "$p") || { echo "PATCH FAILED: $p"; exit 3; }; done

# Every patched file must match the production build byte for byte (patch 09 changes two of them, so skip those).
SKIP=""; [ "${WITH_PATCH_09:-0}" = 1 ] && SKIP=$(targets "$HERE"/patches/vllm/optional/09-*.patch)
bad=0
while read -r sum f; do
  echo "$SKIP" | grep -qx "$f" && continue
  [ "$(sha256sum "$WORK/$f" 2>/dev/null | cut -d' ' -f1)" = "$sum" ] || { echo "CHECKSUM MISMATCH: $f"; bad=1; }
done < "$HERE/EXPECTED.sha256"
[ $bad = 0 ] || { echo "refusing to build: patched files do not match EXPECTED.sha256"; exit 4; }
echo "all patched files match EXPECTED.sha256"
if [ -n "${NCCL_LIB:-}" ]; then mkdir -p "$WORK/$(dirname $NCCL_DST)"; cp "$NCCL_LIB" "$WORK/$NCCL_DST"; FILES="$FILES $NCCL_DST"; fi

{ echo "FROM $BASE"
  for f in $FILES; do echo "COPY $f $SP/$f"; done
  echo "LABEL org.opencontainers.image.description=\"vLLM v0.30.0 + GB10 patches for GLM-5.3-Flash (see README)\""
} > "$WORK/Dockerfile"
docker build -t "$TAG" "$WORK"
echo "built $TAG"
