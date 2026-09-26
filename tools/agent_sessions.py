# Agent-session load at 1, 2, 4 and 8 concurrent sessions: what coding agents actually send, not one-shot prompts.
# Each session runs agent tasks as 3-turn tool loops: system prompt + 6 tools, the model's turn, a canned tool result,
# the next turn. Half the sessions think (enable_thinking true, as coding agents with thinking on send), half do not
# (they send enable_thinking false; the server default in launch_node.sh is thinking on). No temperature is sent, so requests get the model's default
# sampling (1.0 / top_p 0.95), like clients that do not set it. Per level: aggregate completion tok/s over the level's
# wall time, and acceptance from the server's spec-decode counters (foreign traffic is counted, not hidden).
# Usage: ENDPOINT=http://head:8000 python3 agent_sessions.py <label> <out.json> [sweeps, default 3]
import concurrent.futures as cf, json, os, sys, time, urllib.request
B = os.environ.get("ENDPOINT", "http://localhost:8000"); MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash")
LABEL, OUT = (sys.argv + ["test", "/dev/null"])[1:3]; SWEEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LEVELS = {1: 6, 2: 4, 4: 3, 8: 2}  # concurrent sessions: tasks per session (each level takes 1 to 2 minutes)

SYSTEM = """You are an autonomous software engineering agent working in a developer's repository on a Linux workstation.
You can read and edit files, run shell commands, and search the codebase through the tools provided.
Rules:
- Investigate before changing anything: read the relevant files and reproduce the problem when you can.
- Make the smallest change that fixes the root cause. Do not refactor unrelated code.
- Never delete data, force-push, or run destructive commands without being asked.
- After editing, run the relevant tests or a quick check and report the result honestly.
- When you call a tool, call it with complete, valid arguments. Prefer one precise call over several vague ones.
- Keep your final answer short: what was wrong, what you changed, how you verified it.
The repository is a mid-sized Python and TypeScript monorepo: services/ (Python APIs), web/ (TypeScript front end),
scripts/ (ops tooling), config/ (JSON and YAML), and tests/ (pytest). CI runs `make test` and `make lint`."""

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}
S = {"type": "string"}
TOOLS = [
    fn("read_file", "Read a text file, optionally a line range.", {"path": S, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]),
    fn("write_file", "Create or overwrite a file with the given content.", {"path": S, "content": S}, ["path", "content"]),
    fn("edit_file", "Replace an exact string in a file.", {"path": S, "old_string": S, "new_string": S}, ["path", "old_string", "new_string"]),
    fn("run_command", "Run a shell command in the repository root and return its output.", {"command": S, "timeout_s": {"type": "integer"}}, ["command"]),
    fn("grep", "Search file contents with a regular expression.", {"pattern": S, "path": S}, ["pattern"]),
    fn("list_dir", "List a directory.", {"path": S}, ["path"]),
]

TASKS = [
    ("CI job test-api fails with KeyError: 'user_id' in services/auth/session.py. Find the cause and fix it.",
     "def load_session(token, store):\n    raw = store.get(token)\n    if raw is None:\n        return None\n    data = json.loads(raw)\n    return Session(user=data['user_id'], expires=data['exp'])\n\n# store.set() writes {'uid': user.id, 'exp': ts} since commit 4f2a1c9 (rename)\n",
     "tests/test_session.py::test_roundtrip PASSED\ntests/test_session.py::test_expired PASSED\n14 passed in 1.92s"),
    ("Add a --dry-run flag to scripts/cleanup_logs.py that prints what would be deleted without deleting anything.",
     "import argparse, os, time\n\ndef main():\n    p = argparse.ArgumentParser()\n    p.add_argument('--days', type=int, default=14)\n    p.add_argument('--dir', default='/var/log/app')\n    a = p.parse_args()\n    cutoff = time.time() - a.days * 86400\n    for name in os.listdir(a.dir):\n        path = os.path.join(a.dir, name)\n        if os.path.getmtime(path) < cutoff:\n            os.remove(path)\n\nif __name__ == '__main__':\n    main()\n",
     "$ python scripts/cleanup_logs.py --dry-run --days 30\nwould delete /var/log/app/api-2026-08-01.log\nwould delete /var/log/app/api-2026-08-02.log"),
    ("nginx returns 502 for /api/upload when files exceed 10 MB. Diagnose it and propose the exact config change.",
     "server {\n  listen 443 ssl;\n  client_max_body_size 50m;\n  location /api/ {\n    proxy_pass http://api_upstream;\n    proxy_read_timeout 30s;\n    proxy_request_buffering on;\n  }\n}\n# error.log: upstream prematurely closed connection while reading response header, client: 10.0.4.7, request: \"POST /api/upload\"\n# api log: gunicorn worker timeout (30s) on POST /api/upload, body 14.2 MB\n",
     "nginx: the configuration file /etc/nginx/nginx.conf syntax is ok\nnginx: configuration file /etc/nginx/nginx.conf test is successful"),
    ("Write a Python function that reads data/trades.csv and returns realized P&L per symbol (FIFO), with pytest tests.",
     "date,symbol,side,qty,price\n2026-09-01,AAPL,BUY,100,221.40\n2026-09-02,AAPL,SELL,40,226.10\n2026-09-02,MSFT,BUY,50,412.00\n2026-09-03,AAPL,SELL,60,219.80\n2026-09-04,MSFT,SELL,50,418.25\n",
     "tests/test_pnl.py ....  4 passed in 0.31s"),
    ("Summarize what changed in the last 5 commits and tell me whether any of them touch the payment module.",
     "a91c2e0 payments: retry idempotent charges on 409 (services/payments/charge.py)\n7be0d14 web: fix date picker timezone on settings page\n33f8a52 scripts: rotate api logs daily instead of weekly\nd04e9b7 payments: store processor response codes (services/payments/models.py, migration 0042)\n5c1f7aa deps: bump pydantic 2.9 -> 2.11\n",
     "services/payments/charge.py | 38 ++++++++++++++++++-----\nservices/payments/models.py |  6 +++\nmigrations/0042_processor_codes.py | 21 +++++++++++"),
    ("Create config/services/ingest-worker.json for the new ingest-worker service, valid against config/schema.json. Queue ingest.events, 4 workers, batch 500, dead letter queue enabled.",
     "{\n  \"type\": \"object\",\n  \"required\": [\"name\", \"queue\", \"workers\", \"batch_size\", \"retry\", \"dead_letter\", \"metrics\"],\n  \"properties\": {\n    \"name\": {\"type\": \"string\"}, \"queue\": {\"type\": \"string\"}, \"workers\": {\"type\": \"integer\"},\n    \"batch_size\": {\"type\": \"integer\"},\n    \"retry\": {\"type\": \"object\", \"properties\": {\"max_attempts\": {\"type\": \"integer\"}, \"backoff_ms\": {\"type\": \"integer\"}}},\n    \"dead_letter\": {\"type\": \"object\", \"properties\": {\"enabled\": {\"type\": \"boolean\"}, \"queue\": {\"type\": \"string\"}}},\n    \"metrics\": {\"type\": \"object\", \"properties\": {\"port\": {\"type\": \"integer\"}, \"labels\": {\"type\": \"object\"}}}\n  }\n}\n",
     "$ make validate-config\nconfig/services/ingest-worker.json: valid"),
    ("Refactor web/src/utils/dateHelpers.ts to drop the moment.js dependency, keeping behavior identical.",
     "import moment from 'moment';\n\nexport function formatDay(d: Date): string {\n  return moment(d).format('YYYY-MM-DD');\n}\n\nexport function addBusinessDays(d: Date, n: number): Date {\n  let m = moment(d);\n  while (n > 0) {\n    m = m.add(1, 'day');\n    if (m.isoWeekday() < 6) n--;\n  }\n  return m.toDate();\n}\n\nexport function isSameWeek(a: Date, b: Date): boolean {\n  return moment(a).isSame(b, 'isoWeek');\n}\n",
     "PASS src/utils/dateHelpers.test.ts (9 tests)\nTests: 9 passed, 9 total"),
    ("The Docker build for the web image takes 9 minutes. Find what to cache better and change the Dockerfile.",
     "FROM node:22-bookworm\nWORKDIR /app\nCOPY . .\nRUN npm ci\nRUN npm run build\nRUN apt-get update && apt-get install -y curl\nEXPOSE 3000\nCMD [\"npm\", \"start\"]\n",
     "$ docker build -t web .\n[+] Building 41.3s (12/12) FINISHED\n => CACHED [3/8] COPY package*.json ./\n => CACHED [4/8] RUN npm ci"),
]

def post(body):
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

def metrics():
    m = {"acc": 0.0, "draft": 0.0, "req": 0.0, "gen": 0.0}
    for line in urllib.request.urlopen(B + "/metrics", timeout=10).read().decode().splitlines():
        if line.startswith("#") or " " not in line: continue
        name, val = line.rsplit(" ", 1)
        for key, prefix in (("acc", "vllm:spec_decode_num_accepted_tokens_total"), ("draft", "vllm:spec_decode_num_draft_tokens_total"),
                            ("req", "vllm:request_success_total"), ("gen", "vllm:generation_tokens_total")):
            if name.startswith(prefix): m[key] += float(val)
    return m

def task(ti, think, seed):
    user, result_a, result_b = TASKS[ti % len(TASKS)]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]; out = []
    for turn in range(3):
        body = {"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 1536 if think else 1024, "seed": seed + turn,
                "chat_template_kwargs": {"enable_thinking": think}}
        t0 = time.time(); d = post(body); dt = time.time() - t0
        m = d["choices"][0]["message"]; calls = m.get("tool_calls") or []
        out.append({"task": ti, "turn": turn, "think": think, "tokens": d["usage"]["completion_tokens"], "s": round(dt, 2),
                    "finish": d["choices"][0]["finish_reason"], "tool_calls": len(calls)})
        msgs.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": calls} if calls else {})})
        canned = result_a if turn == 0 else result_b
        if calls:
            for i, c in enumerate(calls): msgs.append({"role": "tool", "tool_call_id": c["id"], "content": canned if i == 0 else "ok"})
        else:
            msgs.append({"role": "user", "content": "Here is what the tool returned:\n" + canned + "\nContinue."})
    return out

def level(c, n_tasks, sweep):
    def session(s):
        think = (s % 2 == 0) if c > 1 else None
        res = []
        for k in range(n_tasks):
            th = think if think is not None else (k % 2 == 0)
            res += task(s * 3 + k + sweep, th, seed=1000 * sweep + 100 * s + 10 * k)
        return res
    m0 = metrics(); t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex: reqs = [r for rs in ex.map(session, range(c)) for r in rs]
    wall = time.time() - t0; m1 = metrics(); tok = sum(r["tokens"] for r in reqs)
    return {"c": c, "sweep": sweep, "wall_s": round(wall, 1), "requests": len(reqs), "tokens": tok, "agg_tok_s": round(tok / wall, 1),
            "acceptance": round((m1["acc"] - m0["acc"]) / (m1["draft"] - m0["draft"]), 4) if m1["draft"] > m0["draft"] else None,
            "foreign_requests": int(m1["req"] - m0["req"]) - len(reqs), "foreign_tokens": int(m1["gen"] - m0["gen"]) - tok,
            "tool_call_turns": sum(r["tool_calls"] > 0 for r in reqs), "length_cut": sum(r["finish"] == "length" for r in reqs), "reqs": reqs}

if __name__ == "__main__":
  res = {"label": LABEL, "t0": time.strftime("%FT%T"), "levels": []}
  task(0, False, 1)  # warm-up: fills the prefix cache for the shared system prompt and tools, not measured
  for sweep in range(SWEEPS):
    for c, n in LEVELS.items():
        r = level(c, n, sweep); res["levels"].append(r); json.dump(res, open(OUT, "w"), indent=1)
        print(f"{LABEL} sweep {sweep} C{c}: {r['agg_tok_s']} tok/s ({r['tokens']} tok / {r['wall_s']} s, {r['requests']} req) acc {r['acceptance']} "
              f"tool-call turns {r['tool_call_turns']} length-cut {r['length_cut']} foreign {r['foreign_requests']}/{r['foreign_tokens']}", flush=True)
