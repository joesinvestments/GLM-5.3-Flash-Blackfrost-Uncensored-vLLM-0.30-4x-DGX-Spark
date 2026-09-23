# Drafter A/B harness: identical prompts, temperatures and settings for every arm, acceptance from the server's own
# spec-decode counters (deltas across this run only), foreign traffic detected, greedy outputs hashed so two drafters
# on the same target can be checked for identical text.
# Usage: ENDPOINT=http://head:8000 python3 ab_harness.py <arm_label> <out.json>   (QUICK=1 for a 2-prompt self-test)
import concurrent.futures as cf, hashlib, json, os, random, re, sys, time, urllib.request
B = os.environ.get("ENDPOINT", "http://localhost:8000"); MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash")
ARM, OUT = sys.argv[1], sys.argv[2]
_r = random.Random(3)  # synthetic ~6K-char document for the long-prompt case
CTX = " ".join(f"Item {i}: component {_r.choice('ABCDEFGH')}{_r.randint(10, 99)} passed check {_r.randint(1, 9)} with margin {_r.randint(100, 999)}." for i in range(110))[:6000]
PROMPTS = {
    "code": "Write a Python function that implements binary search on a sorted list, with a docstring and type hints. Then write 4 test cases covering edge cases.",
    "prose": "Write a 250-word short story about a lighthouse keeper who discovers something strange in the fog.",
    "counting": "Count from 1 to 150, comma separated, no words, no explanation.",
    "json": "Return only a JSON array of 6 fictional employees, each with name, role, years_experience and a skills list of 3 items.",
    "explain": "Explain how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery) in about 300 words.",
    "refactor": "Refactor this for readability, add type hints and a docstring, keep behavior identical:\n\ndef f(a,b=None):\n  r=[]\n  for i in range(len(a)):\n    if b is None or a[i] not in b:\n      if a[i] not in r: r.append(a[i])\n  return sorted(r,key=lambda x:(-a.count(x),x))",
    "agentic": "You are a coding agent working in a repo with a CLI tool `cleanlogs` that deletes log files older than N days. Plan the change to add a --dry-run flag in numbered steps, then write the argparse and deletion-loop code.",
    "summary": CTX + "\n\nSummarize the task above in 8 bullet points.",
    "math": "A tank fills at 12 liters per minute and drains at 5 liters per minute. It starts with 40 liters and holds 400. Step by step, how long until it is full? Then check your answer.",
}

def metrics():
    txt = urllib.request.urlopen(B + "/metrics", timeout=10).read().decode()
    m = {"pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"): continue
        v = line.rsplit(" ", 1)
        if len(v) != 2: continue
        name, val = v[0], float(v[1])
        if name.startswith("vllm:spec_decode_num_accepted_tokens_total"): m["acc"] = m.get("acc", 0) + val
        elif name.startswith("vllm:spec_decode_num_draft_tokens_total"): m["draft"] = m.get("draft", 0) + val
        elif name.startswith("vllm:spec_decode_num_drafts_total"): m["drafts"] = m.get("drafts", 0) + val
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            p = re.search(r'position="(\d+)"', name); m["pos"][int(p.group(1))] = m["pos"].get(int(p.group(1)), 0) + val
        elif name.startswith("vllm:request_success_total"): m["req"] = m.get("req", 0) + val
        elif name.startswith("vllm:generation_tokens_total"): m["gen"] = m.get("gen", 0) + val
    return m

def one(prompt, temp, max_tokens=400):
    body = {"model": MODEL, "temperature": temp, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    if temp: body["seed"] = 1234
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); d = json.loads(urllib.request.urlopen(req, timeout=180).read()); dt = time.time() - t0
    text = d["choices"][0]["message"].get("content") or ""
    return d["usage"]["completion_tokens"], dt, text

QUICK = os.environ.get("QUICK") == "1"  # harness self-test: 2 prompts, greedy once, C8 only
if QUICK: PROMPTS = {k: PROMPTS[k] for k in ("code", "counting")}
m0 = metrics(); ours = 0; ours_tok = 0; res = {"arm": ARM, "per_prompt": {}, "t0": time.strftime("%FT%T")}
for label, p in PROMPTS.items():
    runs = []
    for temp in ((0,) if QUICK else (0, 0, 0.7)):
        tok, dt, text = one(p, temp); ours += 1; ours_tok += tok
        runs.append({"temp": temp, "tokens": tok, "s": round(dt, 3), "tok_s": round(tok / dt, 1), "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "head": text[:120]})
    res["per_prompt"][label] = runs
m1 = metrics()
for c in ((8,) if QUICK else (8, 32)):
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        out = list(ex.map(lambda _: one(PROMPTS["code"], 0), range(c)))
    ours += c; wall = time.time() - t0
    res[f"C{c}_tok_s"] = round(sum(o[0] for o in out) / wall, 1)
m2 = metrics()
d_acc, d_draft, d_drafts = m1["acc"] - m0["acc"], m1["draft"] - m0["draft"], m1["drafts"] - m0["drafts"]
res["acceptance_rate"] = round(d_acc / d_draft, 4) if d_draft else None
res["mean_accepted_len"] = round(1 + d_acc / d_drafts, 3) if d_drafts else None
res["per_pos_rate"] = [round((m1["pos"].get(i, 0) - m0["pos"].get(i, 0)) / d_drafts, 3) for i in sorted(m1["pos"])] if d_drafts else []
res["foreign_requests"] = int((m2["req"] - m0["req"]) - ours)
# requests still in flight at m1 do not show in request_success but their tokens DO land in the acceptance counters
res["foreign_tokens_in_acceptance_window"] = int((m1["gen"] - m0["gen"]) - ours_tok)
greedy = {k: [r for r in v if r["temp"] == 0] for k, v in res["per_prompt"].items()}
res["c1_tok_s_mean_greedy"] = {k: round(sum(r["tok_s"] for r in g) / len(g), 1) for k, g in greedy.items()}
res["greedy_deterministic"] = all(len({r["sha"] for r in g}) == 1 for g in greedy.values())
res["greedy_hashes"] = {k: v[0]["sha"] for k, v in res["per_prompt"].items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"ARM {ARM}: acceptance={res['acceptance_rate']} mean_len={res['mean_accepted_len']} per_pos={res['per_pos_rate']} "
      f"C8={res.get('C8_tok_s')} C32={res.get('C32_tok_s')} foreign_req={res['foreign_requests']} foreign_tok={res['foreign_tokens_in_acceptance_window']} greedy_det={res['greedy_deterministic']} "
      f"c1={json.dumps(res['c1_tok_s_mean_greedy'])}", flush=True)
