#!/usr/bin/env python3
"""Thor decode check: the vLLM Thor fork's workload (prose and code prompts, greedy, thinking off, 400 tokens,
median of 3, end-to-end incl. prefill and HTTP), then 1, 2 and 5 concurrent sampled prose requests. Server on :8888."""
import json, time, statistics, urllib.request, concurrent.futures as cf
BASE = "http://127.0.0.1:8888/v1/chat/completions"
PROMPTS = {"prose": "Write a 350-word travel guide to Lisbon.",
           "code": "Write a Python LRU cache class using OrderedDict, with get and put methods, and explain its implementation with example tests."}
def gen(prompt, n=400, greedy=True, seed=None):
    body = {"model": "Qwen3.8-Flash-Next", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
            "chat_template_kwargs": {"enable_thinking": False}}
    if greedy: body.update(temperature=0, top_p=1)
    if seed is not None: body["seed"] = seed
    t = time.monotonic()
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return r, time.monotonic() - t
gen("Reply with OK.", 16)
print("== single stream, greedy, thinking off, 400 tok (vLLM-Thor method; end-to-end incl. prefill+HTTP)")
for name, p in PROMPTS.items():
    runs = []
    for i in range(3):
        r, el = gen(p); tf = r.get("tensorfold", {})
        runs.append(r["usage"]["completion_tokens"] / el)
        print(f"  {name} #{i+1}: {r['usage']['completion_tokens']} tok {el:.2f}s -> {runs[-1]:.1f} tok/s  stats={ {k: tf[k] for k in tf if k in ('decode_s','prefill_s','drafted','accepted','decode_tps')} }")
    print(f"  {name} median: {statistics.median(runs):.2f} tok/s")
print("== concurrency, prose, sampled (seeded), thinking off, 400 tok")
for c in (1, 2, 5):
    t = time.monotonic()
    with cf.ThreadPoolExecutor(c) as ex:
        res = list(ex.map(lambda i: gen(PROMPTS["prose"], greedy=False, seed=1234 + i), range(c)))
    wall = time.monotonic() - t; toks = sum(r["usage"]["completion_tokens"] for r, _ in res)
    print(f"  {c} streams: aggregate {toks/wall:.1f} tok/s, per stream {statistics.mean(r['usage']['completion_tokens']/e for r, e in res):.1f} tok/s")
