import sys, os, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix import *

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
BLOCKS = [(0, 1000), (1000, 1724), (1724, 2626), (2626, 3318)]
tr_df = pd.read_csv(D + "train.csv")
T = load_targets(D + "train_targets.jsonl")
chunk = np.zeros(len(tr_df), dtype=int)
for a, b in BLOCKS:
    for i in range(b - a):
        chunk[a + i] = min(4, int(i * 5 // (b - a)))

nR, nT = len(ARGUMENT_ROLES), len(ENTITY_TYPES)
C = np.ones((nR, nT + 1))
Dm = np.ones((nT, nR + 1))
for i, r in tr_df.iterrows():
    if chunk[i] == 4:
        continue
    t = T[r.id]
    ent = {(e["start"], e["end"]): e["type"] for e in t["entities"]}
    arg = {(a["start"], a["end"]): a["role"] for a in t["arguments"]}
    for sp, role in arg.items():
        j = ARGUMENT_ROLES.index(role)
        C[j, 0 if sp not in ent else 1 + ENTITY_TYPES.index(ent[sp])] += 1
    for sp, ty in ent.items():
        j = ENTITY_TYPES.index(ty)
        Dm[j, 0 if sp not in arg else 1 + ARGUMENT_ROLES.index(arg[sp])] += 1
C = C / C.sum(1, keepdims=True)
Dm = Dm / Dm.sum(1, keepdims=True)
print("P(type|role) rows:")
for j, r in enumerate(ARGUMENT_ROLES):
    print(f"  {r:22s}", " ".join(f"{x:.2f}" for x in C[j]), "  (none," + ",".join(ENTITY_TYPES) + ")")

keys, PA, PE = pickle.load(open("dev/out_mix6.pkl", "rb"))
refs = refs_of(keys)


def couple(pa, pe, alpha):
    fa = (pe @ C.T) ** alpha
    fe = (pa @ Dm.T) ** alpha
    qa = pa.copy()
    qa[:, :, 1:] = pa[:, :, 1:] * fa
    qe = pe.copy()
    qe[:, :, 1:] = pe[:, :, 1:] * fe
    qa = qa / np.maximum(qa.sum(-1, keepdims=True), 1e-12)
    qe = qe / np.maximum(qe.sum(-1, keepdims=True), 1e-12)
    return qa, qe


for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    best = None
    for thr in [0.3, 0.3544, 0.4, 0.45, 0.5]:
        preds = []
        for k in keys:
            qa, qe = couple(PA[k], PE[k], alpha)
            preds.append({"arguments": greedy_prob(qa, "role", ARGUMENT_ROLES, thr), "entities": greedy_prob(qe, "type", ENTITY_TYPES, thr)})
        s = score_docs(preds, refs)
        if best is None or s[0] > best[0]:
            best = (s[0], thr, s[1], s[2])
    print(f"alpha {alpha:.2f} best {best[0]:.4f} at thr {best[1]} (arg {best[2]:.4f} ent {best[3]:.4f})", flush=True)
