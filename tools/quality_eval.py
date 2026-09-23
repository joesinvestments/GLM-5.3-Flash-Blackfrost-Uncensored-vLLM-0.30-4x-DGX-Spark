# Quality check for the KDA recurrent-state dtype A/B (fp32 production vs bf16 arm). Same requests on both arms:
#  A. 30 template-generated multi-step math problems (answers computed here), temperature 0, 8 concurrent.
#  B. 4 needles at 10/40/70/95% depth of a ~38K-token haystack (long prefill: state rounded between prefill chunks).
#  C. 5 long greedy generations (~1500 tokens), each run twice, to compare cross-arm vs within-arm divergence.
# Usage: ENDPOINT=http://head:8000 python3 quality_eval.py <label> <out.json>
import concurrent.futures as cf, json, os, random, re, sys, time, urllib.request
B = os.environ.get("ENDPOINT", "http://localhost:8000") + "/v1/chat/completions"; LABEL, OUT = sys.argv[1], sys.argv[2]
MODEL = os.environ.get("MODEL_NAME", "glm-5.3-flash")

def chat(content, max_tokens):
    body = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(B, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["choices"][0]["message"].get("content") or "", d["usage"]

rng = random.Random(20260923)
def problem(i):
    a, b, c, d = (rng.randint(3, 60) for _ in range(4)); t = i % 6
    if t == 0: c = c % (a * b); return f"A warehouse has {a} crates with {b} bottles each. It ships {c} bottles, then receives {d} more crates of {b} bottles. How many bottles are there now?", a * b - c + d * b
    if t == 1: return f"A cyclist rides at {a} km/h for {b % 7 + 1} hours, then at {c} km/h for {d % 5 + 1} hours. How many kilometers in total?", a * (b % 7 + 1) + c * (d % 5 + 1)
    if t == 2: return f"What is the sum of the first {a} odd numbers, plus {b} times {c}?", a * a + b * c
    if t == 3: return f"A rectangle is {a + 20} by {b + 20}. A square of side {c % 9 + 1} is cut from each of its 4 corners. What area remains?", (a + 20) * (b + 20) - 4 * (c % 9 + 1) ** 2
    if t == 4: return f"Tom has {(a + 50) * 10} dollars. He spends {b} dollars on lunch each day for {c % 6 + 2} days, then earns {d * 3} dollars. How many dollars does he have?", (a + 50) * 10 - b * (c % 6 + 2) + d * 3
    return f"Compute ({a} + {b}) * {c} - {d} * {a % 7 + 2}.", (a + b) * c - d * (a % 7 + 2)
PROBLEMS = [problem(i) for i in range(30)]

WORDS = "river market lantern copper orchard signal harbor meadow engine ledger violet canyon ember glacier quartz summit".split()
def haystack(n_sent=1300):
    h = random.Random(7); s = []
    for i in range(n_sent):
        w = h.sample(WORDS, 5)
        s.append(f"Record {i}: the {w[0]} near the {w[1]} measured {h.randint(100, 999)} units while the {w[2]} and the {w[3]} stayed beside the {w[4]}.")
    return s
NEEDLES = [("Oslo", 0.10, "481937"), ("Lima", 0.40, "705226"), ("Hanoi", 0.70, "339184"), ("Quito", 0.95, "912650")]
def needle_doc():
    s = haystack()
    for city, depth, code in NEEDLES:
        s.insert(int(depth * len(s)), f"IMPORTANT: the secret code for {city} is {code}.")
    return " ".join(s)

LONG = {
    "code": "Write a complete, well-documented Python module implementing an LRU cache with TTL expiry, thread safety, and statistics, followed by a thorough pytest test suite. Be exhaustive.",
    "story": "Write a 1200-word short story about a cartographer who discovers that a coastline on her maps keeps changing overnight. Rich detail, a clear ending.",
    "math": "Solve step by step, showing every intermediate calculation: (1) the number of ways to tile a 2x12 board with 1x2 dominoes, (2) the sum of all primes below 200, (3) 17^13 mod 101, (4) the determinant of the 4x4 matrix [[2,1,0,3],[1,3,2,0],[0,2,4,1],[3,0,1,5]].",
    "explain": "Explain in depth how a modern CPU executes instructions: pipelining, branch prediction, out-of-order execution, caches, and speculative execution vulnerabilities. Aim for about 1200 words.",
    "json": "Produce a JSON array of 40 fictional books, each with title, author, year, genre, page_count and a one-sentence synopsis. Output only JSON.",
}

res = {"label": LABEL, "t0": time.strftime("%FT%T")}
def solve(p):
    text, _ = chat(p[0] + " Show your work briefly, then end with a final line exactly of the form 'ANSWER: <integer>'.", 700)
    m = re.findall(r"ANSWER:\s*(-?\d+)", text)
    return int(m[-1]) if m else None
with cf.ThreadPoolExecutor(8) as ex:
    got = list(ex.map(solve, PROBLEMS))
res["math_correct"] = sum(g == p[1] for g, p in zip(got, PROBLEMS)); res["math_total"] = len(PROBLEMS)
res["math_wrong"] = [i for i, (g, p) in enumerate(zip(got, PROBLEMS)) if g != p[1]]
doc = needle_doc(); nd = []
for city, depth, code in NEEDLES:
    text, usage = chat(doc + f"\n\nQuestion: what is the secret code for {city}? Reply with the 6-digit code only.", 200)
    nd.append({"city": city, "depth": depth, "ok": code in text, "reply": text.strip()[:40], "prompt_tokens": usage["prompt_tokens"]})
res["needles"] = nd; res["needles_correct"] = sum(n["ok"] for n in nd)
long_out = {}
for rnd in (1, 2):
    with cf.ThreadPoolExecutor(len(LONG)) as ex:
        outs = dict(zip(LONG, ex.map(lambda p: chat(p, 1500)[0], LONG.values())))
    for k, v in outs.items(): long_out.setdefault(k, []).append(v)
res["long"] = long_out
def first_diff(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)
res["long_self_divergence_chars"] = {k: first_diff(v[0], v[1]) for k, v in long_out.items()}
json.dump(res, open(OUT, "w"), indent=1)
print(f"QUALITY {LABEL}: math {res['math_correct']}/{res['math_total']} wrong={res['math_wrong']} needles {res['needles_correct']}/4 "
      f"(prompt ~{nd[0]['prompt_tokens']} tokens) self-divergence(chars)={res['long_self_divergence_chars']}", flush=True)
