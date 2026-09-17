import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abgs_metric import ARGUMENT_ROLES, ENTITY_TYPES, ARGUMENT_WEIGHTS, ENTITY_WEIGHTS, match_spans

AW = np.array([ARGUMENT_WEIGHTS[r] for r in ARGUMENT_ROLES])
EW = np.array([ENTITY_WEIGHTS[t] for t in ENTITY_TYPES])


def per_doc_stats(preds, refs):
    na, ne = len(ARGUMENT_ROLES), len(ENTITY_TYPES)
    A = np.zeros((len(preds), na, 4))
    E = np.zeros((len(preds), ne, 4))
    for d, (p, r) in enumerate(zip(preds, refs)):
        for arr, key, field, labels in ((A, "arguments", "role", ARGUMENT_ROLES), (E, "entities", "type", ENTITY_TYPES)):
            for j, lab in enumerate(labels):
                ps = [(x["start"], x["end"]) for x in p[key] if x[field] == lab]
                rs = [(x["start"], x["end"]) for x in r[key] if x[field] == lab]
                soft, exact = match_spans(ps, rs) if ps and rs else (0.0, 0)
                arr[d, j] = (len(ps), len(rs), exact, soft)
    return A, E


def score_from_sums(sa, se):
    def lab(s):
        P, R, Ex, S = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        d = P + R
        with np.errstate(divide="ignore", invalid="ignore"):
            v = 0.7 * (2 * Ex / d) + 0.3 * (2 * S / d)
        v = np.where((P == 0) & (R == 0), 1.0, v)
        v = np.where(((P == 0) ^ (R == 0)), 0.0, v)
        return v
    a = (lab(sa) * AW).sum(-1) / AW.sum()
    e = (lab(se) * EW).sum(-1) / EW.sum()
    return 0.75 * a + 0.25 * e


def paired(stats_base, stats_new, B=2000, seed=0):
    Ab, Eb = stats_base
    An, En = stats_new
    n = Ab.shape[0]
    base = float(score_from_sums(Ab.sum(0), Eb.sum(0)))
    new = float(score_from_sums(An.sum(0), En.sum(0)))
    rng = np.random.RandomState(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        deltas[b] = score_from_sums(An[idx].sum(0), En[idx].sum(0)) - score_from_sums(Ab[idx].sum(0), Eb[idx].sum(0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return dict(base=base, new=new, delta=new - base, lo=float(lo), hi=float(hi), p_gt0=float((deltas > 0).mean()), sd=float(deltas.std()))


def fmt(tag, r):
    sig = "SIGNIFICANT" if (r["lo"] > 0 or r["hi"] < 0) else "not significant"
    return f"{tag:52s} base {r['base']:.4f} new {r['new']:.4f}  delta {r['delta']:+.4f}  95%CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  P(>0) {r['p_gt0']:.2f}  sd {r['sd']:.4f}  {sig}"
