import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix import *
from sklearn.ensemble import HistGradientBoostingClassifier

NAMES = ["xlmrl_crf0_last.npy", "span_xlmrl_last.npy", "indobertweet_crf0_last.npy", "span_ibtw_last.npy",
         "mdeb_crf0_last.npy", "m_mdeb_span_last.npy", "indobertl_crf0_last.npy", "m_ibtl_span_last.npy"]
CAND_MIN = 0.03

mem = [member_spans(n) for n in NAMES]
keys = sorted(mem[0].keys())
refs = refs_of(keys)
PA = {k: np.mean([m[k][0] for m in mem], 0) for k in keys}
PE = {k: np.mean([m[k][1] for m in mem], 0) for k in keys}


def candidates(k, layer):
    P = PA[k] if layer == "arguments" else PE[k]
    Ms = [m[k][0] if layer == "arguments" else m[k][1] for m in mem]
    n = P.shape[0]
    lp = P[:, :, 1:]
    best = lp.argmax(-1)
    bp = np.take_along_axis(lp, best[:, :, None], -1)[:, :, 0]
    srt = np.sort(lp, -1)
    second = srt[:, :, -2] if lp.shape[-1] > 1 else np.zeros_like(bp)
    rows = []
    for s in range(n):
        for l in range(min(LS, n - s)):
            if bp[s, l] <= CAND_MIN:
                continue
            c = int(best[s, l])
            mp = [float(M[s, l, c + 1]) for M in Ms]
            rows.append(dict(key=k, s=s, l=l, c=c, p=float(bp[s, l]), margin=float(bp[s, l] - second[s, l]),
                             pnone=float(P[s, l, 0]), mp=mp, n=n))
    for r in rows:
        ov = [q["p"] for q in rows if q is not r and not (q["s"] + q["l"] < r["s"] or q["s"] > r["s"] + r["l"])]
        r["ov_max"] = max(ov) if ov else 0.0
        r["ov_sum"] = float(sum(ov))
        r["ov_n"] = len(ov)
        same = [q["p"] for q in rows if q["c"] == r["c"]]
        r["rank_in_label"] = int(sum(1 for x in same if x > r["p"]))
        r["best_in_label"] = float(max(same))
    return rows


def featurize(rows):
    X = []
    for r in rows:
        X.append([r["p"], r["margin"], r["pnone"], r["l"] + 1, r["s"] / max(1, r["n"]), (r["s"] + r["l"] + 1) / max(1, r["n"]),
                  r["n"], r["ov_max"], r["ov_sum"], r["ov_n"], r["rank_in_label"], r["best_in_label"],
                  r["p"] / max(1e-6, r["best_in_label"]), float(np.std(r["mp"])), float(np.min(r["mp"])), float(np.max(r["mp"])), r["c"]] + r["mp"])
    return np.array(X, dtype=np.float64)


def gold_set(k, layer, field):
    return {(x[field], x["start"], x["end"]) for x in refs[keys.index(k)][layer]}


def decode_scored(rows, scores, labels, field, thr):
    by_doc = {}
    for r, sc in zip(rows, scores):
        by_doc.setdefault(r["key"], []).append((sc, r))
    out = {}
    for k, items in by_doc.items():
        items.sort(key=lambda x: -x[0])
        n = items[0][1]["n"]
        used = np.zeros(n, dtype=bool)
        sp = []
        for sc, r in items:
            if sc <= thr:
                continue
            if used[r["s"]: r["s"] + r["l"] + 1].any():
                continue
            used[r["s"]: r["s"] + r["l"] + 1] = True
            sp.append({field: labels[r["c"]], "start": r["s"], "end": r["s"] + r["l"] + 1})
        out[k] = sorted(sp, key=lambda x: x["start"])
    return out


cand = {layer: {k: candidates(k, layer) for k in keys} for layer in ("arguments", "entities")}
print("candidates per doc:", {l: round(np.mean([len(cand[l][k]) for k in keys]), 1) for l in cand})

rng = np.random.RandomState(0)
perm = rng.permutation(len(keys))
halves = [[keys[i] for i in perm[: len(keys) // 2]], [keys[i] for i in perm[len(keys) // 2:]]]

base_pred = {k: {"arguments": greedy_prob(PA[k], "role", ARGUMENT_ROLES, 0.3544), "entities": greedy_prob(PE[k], "type", ENTITY_TYPES, 0.3544)} for k in keys}
rr_pred = {k: {} for k in keys}

for layer, labels, field in (("arguments", ARGUMENT_ROLES, "role"), ("entities", ENTITY_TYPES, "type")):
    for tr_keys, te_keys in ((halves[0], halves[1]), (halves[1], halves[0])):
        rows_tr = [r for k in tr_keys for r in cand[layer][k]]
        y_tr = np.array([1 if (labels[r["c"]], r["s"], r["s"] + r["l"] + 1) in gold_set(r["key"], layer, field) else 0 for r in rows_tr])
        X_tr = featurize(rows_tr)
        groups = np.array([tr_keys.index(r["key"]) % 2 for r in rows_tr])
        s_tr = np.zeros(len(rows_tr))
        for g in (0, 1):
            inner = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            inner.fit(X_tr[groups != g], y_tr[groups != g])
            s_tr[groups == g] = inner.predict_proba(X_tr[groups == g])[:, 1]
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
        model.fit(X_tr, y_tr)
        best_thr, best_v = 0.3, -1
        for thr in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            dec = decode_scored(rows_tr, s_tr, labels, field, thr)
            tp = sum(len({(x[field], x["start"], x["end"]) for x in dec[k]} & gold_set(k, layer, field)) for k in tr_keys)
            npred = sum(len(dec[k]) for k in tr_keys)
            ngold = sum(len(gold_set(k, layer, field)) for k in tr_keys)
            v = 2 * tp / max(1, npred + ngold)
            if v > best_v:
                best_v, best_thr = v, thr
        rows_te = [r for k in te_keys for r in cand[layer][k]]
        s_te = model.predict_proba(featurize(rows_te))[:, 1]
        dec = decode_scored(rows_te, s_te, labels, field, best_thr)
        for k in te_keys:
            rr_pred[k][layer] = dec.get(k, [])
        print(f"  {layer} fold thr {best_thr} train F1 {best_v:.4f}", flush=True)

print("baseline    %.4f %.4f %.4f" % score_docs([base_pred[k] for k in keys], refs))
print("reranked    %.4f %.4f %.4f" % score_docs([rr_pred[k] for k in keys], refs))
