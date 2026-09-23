#!/usr/bin/env bash
# Safe lab lane: the ONLY way an experiment (or an AI agent) touches a cluster that is also serving production.
# Runs one command on all 4 nodes in parallel, from ./src (synced to each node), with hard safety gates:
#   - before AND after: production must answer a REAL completion (a 200 from /health is not enough) and every node
#     must have >= LAB_MARGIN_MIB above the earlyoom SIGTERM line;
#   - container mode additionally requires production to be idle (no running or waiting requests) before it starts;
#   - every rank runs under oomwrap with a floor LAB_FLOOR_MIB above the earlyoom line, so a runaway experiment is
#     killed long before earlyoom picks the vLLM worker;
#   - one run at a time (lock), timeouts capped, outputs in results/<label>/rank<N>.log (last line: exit=<code>).
# Usage:
#   lab_run.sh host      <label> <timeout_s<=900> '<command>'   host shell, cwd = synced src/
#   lab_run.sh container <label> <timeout_s<=900> '<command>'   throwaway container from LAB_IMAGE with GPU + RDMA
# Each rank sees RANK (0-3), WORLD=4, IPS (rail IPs), MASTER_ADDR (rank 0 rail IP), MASTER_PORT.
# Exit: 0 ok, 2 refused by a gate, 3 production unhealthy AFTER the run (stop everything), 4 failure/timeout.
# Required env: LAB_HOSTS (4 ssh names, rank order), LAB_RAIL_IPS (4 IPs), LAB_PROD_URL (e.g. http://head:8000).
# The body is one { } block so bash parses all of it before running: editing this file mid-run cannot corrupt a live run.
{
set -u
MODE=${1:?mode host|container}; LABEL=${2:?label}; T=${3:?timeout_s}; CMD=${4:?command}
HERE=$(cd "$(dirname "$0")" && pwd)
read -r -a HOSTS <<< "${LAB_HOSTS:?set LAB_HOSTS to the 4 ssh host names, rank 0 first}"
IPS=${LAB_RAIL_IPS:?set LAB_RAIL_IPS to the 4 rail IPs, rank 0 first}; read -r -a IPA <<< "$IPS"
PROD=${LAB_PROD_URL:?set LAB_PROD_URL, e.g. http://head:8000}; MODEL=${LAB_MODEL_NAME:-glm-5.3-flash}
IMAGE=${LAB_IMAGE:-glm53-flash-gb10:v0.30.0}; REMOTE=${LAB_REMOTE_DIR:-/var/tmp/lab}; MPORT=${LAB_MASTER_PORT:-29700}
EARLYOOM_MIB=${LAB_EARLYOOM_MIB:-4982}      # earlyoom -m 4 on a 124,546 MiB Spark = SIGTERM below ~4,982 MiB available
MARGIN_MIB=${LAB_MARGIN_MIB:-3072}; FLOOR_MIB=$((EARLYOOM_MIB + ${LAB_FLOOR_MIB:-2560}))
OOMWRAP=${LAB_OOMWRAP:-'~/.cargo/bin/oomwrap'}   # quoted: the tilde must expand on each node, not locally
RAIL_HCAS=${RAIL_HCAS:-rocep1s0f0,roceP2p1s0f0}; RAIL_IFS=${RAIL_IFS:-enp1s0f0np0,enP2p1s0f0np0}; GLOO_IF=${GLOO_IF:-enp1s0f0np0}; GID_INDEX=${GID_INDEX:-3}
SSH="/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=8"   # absolute path on purpose, see README "agent guard"
OOMW="$OOMWRAP run --min-mem ${FLOOR_MIB}M --min-swap 1G --poll 200ms --event-log $REMOTE/oomwrap-events.jsonl"
[ ${#HOSTS[@]} = 4 ] && [ ${#IPA[@]} = 4 ] || { echo "LAB_HOSTS and LAB_RAIL_IPS must each list 4 entries"; exit 4; }
[[ "$MODE" =~ ^(host|container)$ ]] || { echo "mode must be host or container"; exit 4; }
[[ "$LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "bad label"; exit 4; }
[ "$T" -le 900 ] 2>/dev/null || { echo "timeout must be <= 900s"; exit 4; }
echo "$(date +%FT%T) $MODE $LABEL agent-guard=$(declare -F ssh >/dev/null && echo on || echo off)" >> "$HERE/.lab_run.history"

LOCK="$HERE/.lab.lock"; waited=0
until mkdir "$LOCK" 2>/dev/null; do
  holder=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null; continue; fi
  [ $waited -ge 1200 ] && { echo "REFUSED: another lab_run (pid $holder) held the lock for 20 min"; exit 2; }
  [ $waited = 0 ] && echo "waiting for lab lock held by pid $holder"
  sleep 5; waited=$((waited + 5))
done
echo $$ > "$LOCK/pid"; trap 'rm -f "$LOCK/pid"; rmdir "$LOCK"' EXIT

gate() {
  local code; code=$(curl -s -o /dev/null -w "%{http_code}" -m 60 "$PROD/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}")
  [ "$code" = 200 ] || { echo "GATE: production real completion returned $code"; return 1; }
  if [ "$MODE" = container ] && [ "${1:-}" = pre ]; then   # a torch process landing during a big prefill is how earlyoom killed production
    local busy; busy=$(curl -s -m 8 "$PROD/metrics" | awk '/^vllm:num_requests_(running|waiting)\{/{s+=$NF} END{print s+0}')
    [ "${busy%.*}" = 0 ] || { echo "GATE: production has $busy requests in flight; container mode needs it idle"; return 1; }
  fi
  for h in "${HOSTS[@]}"; do
    local m; m=$($SSH "$h" "awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo") || { echo "GATE: $h unreachable"; return 1; }
    [ $((m - EARLYOOM_MIB)) -ge $MARGIN_MIB ] || { echo "GATE: $h only $((m - EARLYOOM_MIB)) MiB above earlyoom"; return 1; }
  done
}
gate pre || { echo "REFUSED: pre-run gate failed"; exit 2; }

OUT="$HERE/results/$LABEL"; mkdir -p "$OUT"
for h in "${HOSTS[@]}"; do
  $SSH "$h" "mkdir -p $REMOTE" && command rsync -a --delete --exclude bin/ -e "$SSH" "$HERE/src/" "$h:$REMOTE/src/" \
    && command rsync -a -e "$SSH" "$HERE/oomwrap_supervise.sh" "$h:$REMOTE/" || { echo "sync to $h failed"; exit 4; }
done

NCCL_ENV="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$RAIL_HCAS -e NCCL_SOCKET_IFNAME=$RAIL_IFS -e GLOO_SOCKET_IFNAME=$GLOO_IF -e NCCL_IB_GID_INDEX=$GID_INDEX -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=1 -e NCCL_MIN_NCHANNELS=4 -e NCCL_MAX_NCHANNELS=4 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN"
pids=()
for r in 0 1 2 3; do
  h=${HOSTS[$r]}
  if [ "$MODE" = host ]; then
    $SSH "$h" "cd $REMOTE/src && export RANK=$r WORLD=4 IPS='$IPS' MASTER_ADDR=${IPA[0]} MASTER_PORT=$MPORT && \
      timeout $T $OOMW --profile generic --term-grace 1s -- bash -c $(printf %q "$CMD"); echo \"[lab_run] rank $r exit=\$?\"" > "$OUT/rank$r.log" 2>&1 &
  else
    N="lab-$LABEL-r$r"
    $SSH "$h" "docker rm -f $N >/dev/null 2>&1; timeout $((T + 30)) $OOMW --profile vllm --term-grace 2s -- bash $REMOTE/oomwrap_supervise.sh $N \
      docker run -d --pull never --name $N --gpus all --network host --ipc host --memory 8g --device /dev/infiniband --cap-add IPC_LOCK \
      --ulimit memlock=-1:-1 $NCCL_ENV -e RANK=$r -e WORLD=4 -e WORLD_SIZE=4 -e IPS='$IPS' -e MASTER_ADDR=${IPA[0]} -e MASTER_PORT=$MPORT \
      -v $REMOTE:/lab --entrypoint bash $IMAGE -c $(printf %q "cd /lab/src && timeout $T bash -c $(printf %q "$CMD")") >/dev/null;
      sup=\$?; rc=\$(docker inspect -f '{{.State.ExitCode}}' $N 2>/dev/null || echo 99); [ \$sup -ne 0 ] && rc=\$sup;
      docker logs $N 2>&1; docker kill $N >/dev/null 2>&1; docker rm -f $N >/dev/null 2>&1; echo \"[lab_run] rank $r exit=\$rc\"" > "$OUT/rank$r.log" 2>&1 &
  fi
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=4; done
for r in 0 1 2 3; do tail -1 "$OUT/rank$r.log" | grep -q "exit=0$" || rc=4; done
gate || { echo "WARNING: production unhealthy AFTER the run. Stop all lab work and report."; exit 3; }
echo "lab_run $MODE $LABEL finished (rc=$rc); outputs in $OUT"
exit $rc
}
