import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ens import *

O = "colab_out/out/"
LS = 20


def bio_to_span(lp, tags, labels):
    p = np.exp(lp.astype(np.float64))
    n = p.shape[0]
    K = len(labels)
    out = np.zeros((n, LS, K + 1))
    for k, lab in enumerate(labels):
        pb = p[:, tags.index("B-" + lab)]
        pi = p[:, tags.index("I-" + lab)]
        cont = np.concatenate([pi, [0.0]])
        for s in range(n):
            acc = pb[s]
            for l in range(min(LS, n - s)):
                if l > 0:
                    acc = acc * pi[s + l]
                out[s, l, k + 1] = acc * (1.0 - cont[s + l + 1])
    out[:, :, 0] = np.maximum(0.0, 1.0 - out[:, :, 1:].sum(-1))
    return out


def greedy_prob(P, key, labels, thr, bias=None):
    n, L, K1 = P.shape
    Q = P[:, :, 1:] if bias is None else P[:, :, 1:] * bias[None, None, :]
    best = Q.argmax(-1)
    bp = np.take_along_axis(Q, best[:, :, None], -1)[:, :, 0]
    cand = [(bp[s, l], s, l, best[s, l]) for s in range(n) for l in range(min(L, n - s)) if bp[s, l] > thr]
    cand.sort(key=lambda x: -x[0])
    used = np.zeros(n, dtype=bool)
    out = []
    for sc, s, l, c in cand:
        if not used[s: s + l + 1].any():
            used[s: s + l + 1] = True
            out.append({key: labels[c], "start": int(s), "end": int(s + l + 1)})
    return sorted(out, key=lambda x: x["start"])


def member_spans(name):
    d = np.load(O + name, allow_pickle=True).item()
    keys = sorted(d.keys())
    out = {}
    for k in keys:
        a, e = d[k]
        if a.ndim == 3:
            pa = np.zeros((a.shape[0], LS, a.shape[2])); pa[:, : a.shape[1]] = np.exp(a.astype(np.float64))
            pe = np.zeros((e.shape[0], LS, e.shape[2])); pe[:, : e.shape[1]] = np.exp(e.astype(np.float64))
            out[k] = (pa, pe)
        else:
            out[k] = (bio_to_span(a, ARG_TAGS, ARGUMENT_ROLES), bio_to_span(e, ENT_TAGS, ENTITY_TYPES))
    return out


if __name__ == "__main__":
    names = sys.argv[1].split(",")
    thrs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.3, 0.4, 0.5]
    mem = {n: member_spans(n) for n in names}
    keys = sorted(mem[names[0]].keys())
    refs = refs_of(keys)
    for n in names:
        for thr in [0.5]:
            preds = [{"arguments": greedy_prob(mem[n][k][0], "role", ARGUMENT_ROLES, thr), "entities": greedy_prob(mem[n][k][1], "type", ENTITY_TYPES, thr)} for k in keys]
            print(n, thr, "%.4f %.4f %.4f" % score_docs(preds, refs), flush=True)
    for thr in thrs:
        preds = []
        for k in keys:
            PA = np.mean([mem[n][k][0] for n in names], 0)
            PE = np.mean([mem[n][k][1] for n in names], 0)
            preds.append({"arguments": greedy_prob(PA, "role", ARGUMENT_ROLES, thr), "entities": greedy_prob(PE, "type", ENTITY_TYPES, thr)})
        print("MIX", len(names), thr, "%.4f %.4f %.4f" % score_docs(preds, refs), flush=True)
