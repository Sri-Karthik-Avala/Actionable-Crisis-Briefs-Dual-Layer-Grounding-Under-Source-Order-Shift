import json, collections, sys
import numpy as np
import pandas as pd
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from abgs_metric import *

D = "C:/Users/srika/Downloads/eris_actionable/"
tr = pd.read_csv(D + "train.csv")
te = pd.read_csv(D + "test.csv")
T = load_targets(D + "train_targets.jsonl")
print(tr.shape, te.shape, len(T))
print("n_tokens train", tr.n_tokens.describe().to_dict())
print("n_tokens test", te.n_tokens.describe().to_dict())
assert all(len(t.split(" ")) == n for t, n in zip(tr.text, tr.n_tokens)), "split(' ') mismatch"
bad = sum(len(t.split()) != n for t, n in zip(tr.text, tr.n_tokens))
print("split() mismatches", bad)

refs = [T[i] for i in tr.id]
ac = collections.Counter(); ec = collections.Counter()
alen = collections.defaultdict(list); elen = collections.defaultdict(list)
docs_with = collections.Counter()
for r in refs:
    seen = set()
    for a in r["arguments"]:
        ac[a["role"]] += 1; alen[a["role"]].append(a["end"] - a["start"]); seen.add(a["role"])
    for e in r["entities"]:
        ec[e["type"]] += 1; elen[e["type"]].append(e["end"] - e["start"])
    for s in seen: docs_with[s] += 1
print("\nARG role counts / docs / len mean / len max / len hist")
for k in ARGUMENT_ROLES:
    L = np.array(alen[k]) if alen[k] else np.array([0])
    print(f"  {k:22s} n={ac[k]:5d} docs={docs_with[k]:5d} mean={L.mean():.2f} p90={np.percentile(L,90):.0f} max={L.max()}")
print("ENT")
for k in ENTITY_TYPES:
    L = np.array(elen[k]) if elen[k] else np.array([0])
    print(f"  {k:22s} n={ec[k]:5d} mean={L.mean():.2f} p90={np.percentile(L,90):.0f} max={L.max()}")

same = 0; tot = 0
pair = collections.Counter()
ent_only = collections.Counter(); arg_only = collections.Counter()
for r in refs:
    es = {(e["start"], e["end"]): e["type"] for e in r["entities"]}
    as_ = {(a["start"], a["end"]): a["role"] for a in r["arguments"]}
    for sp, role in as_.items():
        tot += 1
        if sp in es:
            same += 1; pair[(role, es[sp])] += 1
        else:
            arg_only[role] += 1
    for sp, t in es.items():
        if sp not in as_:
            ent_only[t] += 1
print(f"\narg spans with identical entity span: {same}/{tot}")
print("entity-only spans", dict(ent_only))
print("arg-only spans", dict(arg_only))
print("role -> type table")
tab = collections.defaultdict(collections.Counter)
for (r_, t_), c in pair.items(): tab[r_][t_] += c
for r_ in ARGUMENT_ROLES:
    print(f"  {r_:22s}", dict(tab[r_]))

evt = []
for r in refs:
    ev = [a["role"] for a in r["arguments"] if a["role"].endswith("-EVENT") and a["role"] != "FALSE-EVENT"]
    evt.append(collections.Counter(ev).most_common(1)[0][0] if ev else ("FALSE" if any(a["role"] == "FALSE-EVENT" for a in r["arguments"]) else "NONE"))
tr["evt"] = evt
print("\ndoc event type counts", collections.Counter(evt))
print("first 40 row event types:", evt[:40])
runs = sum(1 for i in range(1, len(evt)) if evt[i] != evt[i - 1])
print("event-type changes along row order", runs, "of", len(evt) - 1)
print("empty docs (no args)", sum(1 for r in refs if not r["arguments"]), "no ents", sum(1 for r in refs if not r["entities"]))
mixed = sum(1 for r in refs if len({a["role"] for a in r["arguments"] if a["role"].endswith("-EVENT") and a["role"] != "FALSE-EVENT"}) > 1)
print("docs with >1 real event type", mixed)

vc = tr.text.value_counts()
print("\nduplicate texts in train", (vc > 1).sum(), "rows", vc[vc > 1].sum())
trs = set(tr.text)
print("test texts exactly in train", sum(t in trs for t in te.text))
kw = {"kebakaran": 0, "banjir": 0, "gempa": 0, "kecelakaan": 0, "tabrakan": 0, "terbakar": 0}
for name, df in [("train", tr), ("test", te)]:
    c = {k: int(df.text.str.contains(k).sum()) for k in kw}
    print(name, c)

sample = pd.read_csv(D + "sample_submission.csv")
sp = {i: json.loads(p) for i, p in zip(sample.id, sample.prediction)}
print("sample ids == test ids in order", list(sample.id) == list(te.id))
print("self-score", score_docs(refs, refs))
empty = [{"n_tokens": r["n_tokens"], "arguments": [], "entities": []} for r in refs]
print("empty score", score_docs(empty, refs))
