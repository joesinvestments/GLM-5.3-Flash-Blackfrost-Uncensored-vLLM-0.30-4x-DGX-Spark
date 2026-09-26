#!/usr/bin/env bash
# Build the production image: the vLLM main nightly (commit 7f1a5398e9) + our patch series + the FlashInfer GB10 fixes.
# Copies the target files out of the base image, applies patches/vllm/series in order and the FlashInfer patches, checks every
# result against EXPECTED.sha256 (the files of the image we serve), then builds a COPY-only image (no container runs).
#   ./build.sh                                  -> glm53-flash-gb10:main-7f1a
#   TAG=myname:tag ./build.sh                   -> custom tag
#   NCCL_LIB=nccl/libnccl.so.2 ./build.sh       -> also replace the bundled NCCL 2.30.7 with the fence-fixed build (nccl/build_nccl.sh)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423}
TAG=${TAG:-glm53-flash-gb10:main-7f1a}
SP=/usr/local/lib/python3.12/dist-packages
NCCL_DST=nvidia/nccl/lib/libnccl.so.2
mapfile -t VP < <(grep -v '^\s*$' "$HERE/patches/vllm/series" | sed "s#^#$HERE/patches/vllm/#")
PATCHES=("${VP[@]}" "$HERE"/patches/flashinfer/*.patch)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"; docker rm -f glm53-gb10-extract >/dev/null 2>&1 || true' EXIT
FILES=$(grep -h '^+++ b/' "${PATCHES[@]}" | sed 's#^+++ b/##' | cut -f1 | sort -u)

docker image inspect "$BASE" >/dev/null 2>&1 || { echo "base image $BASE not present; pull it first (about 20 GB)"; exit 2; }
docker create --name glm53-gb10-extract "$BASE" >/dev/null
for f in $FILES; do  # some patches add new files, which the base image does not have
  mkdir -p "$WORK/$(dirname "$f")"; docker cp "glm53-gb10-extract:$SP/$f" "$WORK/$f" 2>/dev/null || echo "new file: $f"
done
for p in "${PATCHES[@]}"; do (cd "$WORK" && patch -p1 -F0 --forward --no-backup-if-mismatch -s < "$p") || { echo "PATCH FAILED: $p"; exit 3; }; done

# Every patched file must match the production build byte for byte.
bad=0
while read -r sum f; do
  [ "$(sha256sum "$WORK/$f" 2>/dev/null | cut -d' ' -f1)" = "$sum" ] || { echo "CHECKSUM MISMATCH: $f"; bad=1; }
done < "$HERE/EXPECTED.sha256"
[ "$(wc -l < "$HERE/EXPECTED.sha256")" = "$(echo "$FILES" | wc -l)" ] || { echo "EXPECTED.sha256 does not list every patched file"; bad=1; }
[ $bad = 0 ] || { echo "refusing to build: patched files do not match EXPECTED.sha256"; exit 4; }
echo "all $(echo "$FILES" | wc -l) patched files match EXPECTED.sha256"
if [ -n "${NCCL_LIB:-}" ]; then mkdir -p "$WORK/$(dirname $NCCL_DST)"; cp "$NCCL_LIB" "$WORK/$NCCL_DST"; FILES="$FILES $NCCL_DST"; fi

{ echo "FROM $BASE"
  for f in $FILES; do echo "COPY $f $SP/$f"; done
  echo "LABEL org.opencontainers.image.description=\"vLLM main 7f1a + GB10 patches for GLM-5.3-Flash (see README)\""
} > "$WORK/Dockerfile"
docker build -t "$TAG" "$WORK"
echo "built $TAG"
