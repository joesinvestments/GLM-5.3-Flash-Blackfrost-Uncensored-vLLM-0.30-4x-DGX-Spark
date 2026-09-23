#!/usr/bin/env bash
# Lets oomwrap (github.com/osolmaz/oomwrap) supervise a detached container: `docker run -d` returns at once, so this
# runs it and then blocks on `docker wait`. On SIGTERM/SIGINT (oomwrap's memory-pressure kill or an external stop) it
# uses `docker kill`, not `docker stop`: a graceful stop cannot finish inside oomwrap's --term-grace window unless the
# container's PID 1 exits on SIGTERM immediately, and under memory pressure reclaiming memory fast is the point.
# Usage: oomwrap run ... -- bash oomwrap_supervise.sh <container-name> docker run -d --name <container-name> ...
set -uo pipefail
NAME="$1"; shift
cleanup() {
  echo "[oomwrap_supervise] signal received, killing container $NAME" >&2
  docker kill "$NAME" >/dev/null 2>&1
  exit 137
}
trap cleanup TERM INT
"$@"
rc=$?
if [ $rc -ne 0 ]; then
  echo "[oomwrap_supervise] docker run failed to start $NAME (exit $rc)" >&2
  exit $rc
fi
docker wait "$NAME"
