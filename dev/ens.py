import sys, os, glob, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abgs_metric import ARGUMENT_ROLES, ENTITY_TYPES, score_docs, load_targets

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
ARG_TAGS = ["O"] + [p + r for r in ARGUMENT_ROLES for p in ("B-", "I-")]
ENT_TAGS = ["O"] + [p + t for t in ENTITY_TYPES for p in ("B-", "I-")]

tr = pd.read_csv(D + "train.csv")
T = load_targets(D + "train_targets.jsonl")


def allowed(tags):
    K = len(tags)
    M = np.zeros((K, K), dtype=bool)
    st = np.zeros(K, dtype=bool)
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            M[i, j] = (a != "O" and a[2:] == b[2:]) if b.startswith("I-") else True
        st[i] = not a.startswith("I-")
    return np.where(M, 0.0, -1e9), np.where(st, 0.0, -1e9)


AM = allowed(ARG_TAGS)
EM = allowed(ENT_TAGS)


def viterbi(lp, masks):
    tm, sm = masks
    n, K = lp.shape
    dp = sm + lp[0]
    bp = np.zeros((n, K), dtype=int)
    for t in range(1, n):
        c = dp[:, None] + tm
        bp[t] = c.argmax(0)
        dp = c.max(0) + lp[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return path[::-1]


def spans_of(path, tags, key):
    out = []
    cur = None
    for i in range(len(path) + 1):
        t = tags[path[i]] if i < len(path) else "O"
        if cur is not None and not (t.startswith("I-") and t[2:] == cur[0]):
            out.append({key: cur[0], "start": cur[1], "end": i})
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


def load(paths, mode="logp"):
    dumps = [np.load(p, allow_pickle=True).item() for p in paths]
    keys = sorted(dumps[0].keys())
    out = {}
    for k in keys:
        if mode == "logp":
            la = np.mean([d[k][0].astype(np.float32) for d in dumps], 0)
            le = np.mean([d[k][1].astype(np.float32) for d in dumps], 0)
        else:
            la = np.log(np.mean([np.exp(d[k][0].astype(np.float32)) for d in dumps], 0) + 1e-9)
            le = np.log(np.mean([np.exp(d[k][1].astype(np.float32)) for d in dumps], 0) + 1e-9)
        out[k] = (la, le)
    return out


def decode(res, keys, ba=None, be=None):
    preds = []
    for k in keys:
        la, le = res[k]
        if ba is not None:
            la = la + ba
        if be is not None:
            le = le + be
        preds.append({"arguments": spans_of(viterbi(la, AM), ARG_TAGS, "role"), "entities": spans_of(viterbi(le, EM), ENT_TAGS, "type")})
    return preds


EVR = ["FIRE-EVENT", "FLOOD-EVENT", "EARTHQUAKE-EVENT", "ACCIDENT-EVENT", "FALSE-EVENT"]


def event_consistent(res, keys, preds):
    out = []
    for k, p in zip(keys, preds):
        la = res[k][0]
        ev = [a for a in p["arguments"] if a["role"] in EVR]
        if len(ev) > 1:
            tot = np.zeros(len(EVR))
            for a in ev:
                for j, r in enumerate(EVR):
                    pr = np.exp(la[a["start"]:a["end"], [ARG_TAGS.index("B-" + r), ARG_TAGS.index("I-" + r)]]).sum(1)
                    tot[j] += np.log(pr + 1e-6).sum()
            best = EVR[int(tot.argmax())]
            args_ = [dict(a, role=best) if a["role"] in EVR else a for a in p["arguments"]]
            p = dict(p, arguments=args_)
        out.append(p)
    return out


def cross_fill(preds):
    out = []
    for p in preds:
        ents = list(p["entities"])
        occ = set()
        for e in ents:
            occ |= set(range(e["start"], e["end"]))
        for a in p["arguments"]:
            if a["role"] in EVR:
                t = "EVE"
            elif a["role"] in ("PLACE-ARG",):
                continue
            elif a["role"] == "OFFICER-ARG":
                t = "ORG"
            else:
                t = "ARG"
            if not (set(range(a["start"], a["end"])) & occ):
                ents.append({"type": t, "start": a["start"], "end": a["end"]})
                occ |= set(range(a["start"], a["end"]))
        out.append(dict(p, entities=ents))
    return out


def refs_of(keys):
    return [T[tr.id[k]] for k in keys]


def bias_vec(tags, labs, vals):
    b = np.zeros(len(tags), dtype=np.float32)
    for lab, v in zip(labs, vals):
        b[tags.index("B-" + lab)] += v
        b[tags.index("I-" + lab)] += v
    return b


def tune_bias(res, keys, grid=(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)):
    refs = refs_of(keys)
    va = np.zeros(len(ARGUMENT_ROLES)); ve = np.zeros(len(ENTITY_TYPES))
    for it in range(2):
        for j, lab in enumerate(ARGUMENT_ROLES):
            best = None
            for g in grid:
                v = va.copy(); v[j] = g
                s = score_docs(decode(res, keys, bias_vec(ARG_TAGS, ARGUMENT_ROLES, v), bias_vec(ENT_TAGS, ENTITY_TYPES, ve)), refs)[0]
                if best is None or s > best[0] + 1e-9:
                    best = (s, g)
            va[j] = best[1]
        for j, lab in enumerate(ENTITY_TYPES):
            best = None
            for g in grid:
                v = ve.copy(); v[j] = g
                s = score_docs(decode(res, keys, bias_vec(ARG_TAGS, ARGUMENT_ROLES, va), bias_vec(ENT_TAGS, ENTITY_TYPES, v)), refs)[0]
                if best is None or s > best[0] + 1e-9:
                    best = (s, g)
            ve[j] = best[1]
    return va, ve


if __name__ == "__main__":
    paths = sys.argv[1:]
    res = load(paths)
    keys = sorted(res.keys())
    refs = refs_of(keys)
    base = score_docs(decode(res, keys), refs, verbose=True)
    print("ensemble", len(paths), "score", base)
    p0 = decode(res, keys)
    print("event-consistent", score_docs(event_consistent(res, keys, p0), refs))
    print("cross-fill ents", score_docs(cross_fill(p0), refs))
    rng = np.random.RandomState(0)
    perm = rng.permutation(keys)
    h1, h2 = sorted(perm[: len(perm) // 2]), sorted(perm[len(perm) // 2:])
    gains = []
    for a, b in ((h1, h2), (h2, h1)):
        va, ve = tune_bias(res, a)
        s0 = score_docs(decode(res, b), refs_of(b))[0]
        s1 = score_docs(decode(res, b, bias_vec(ARG_TAGS, ARGUMENT_ROLES, va), bias_vec(ENT_TAGS, ENTITY_TYPES, ve)), refs_of(b))[0]
        gains.append(s1 - s0)
        print("crossfit bias", dict(zip(ARGUMENT_ROLES, va)), dict(zip(ENTITY_TYPES, ve)), f"{s0:.4f} -> {s1:.4f}")
    print("mean crossfit bias gain", np.mean(gains))
