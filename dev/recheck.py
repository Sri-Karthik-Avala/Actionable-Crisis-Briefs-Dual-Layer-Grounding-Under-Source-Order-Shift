import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mix import *
from paired import per_doc_stats, paired, fmt

P = {"xb": "xlmrl_crf0_last.npy", "xs": "span_xlmrl_last.npy", "wb": "indobertweet_crf0_last.npy", "ws": "span_ibtw_last.npy",
     "db": "mdeb_crf0_last.npy", "ds": "m_mdeb_span_last.npy", "lb": "indobertl_crf0_last.npy", "ls": "m_ibtl_span_last.npy"}
cache = {}


def get(n):
    if n not in cache:
        cache[n] = member_spans(P[n])
    return cache[n]


keys = sorted(get("xb").keys())
refs = refs_of(keys)


def preds_for(names, thr=0.3544):
    mem = [get(n) for n in names]
    out = []
    for k in keys:
        pa = np.mean([m[k][0] for m in mem], 0)
        pe = np.mean([m[k][1] for m in mem], 0)
        out.append({"arguments": greedy_prob(pa, "role", ARGUMENT_ROLES, thr), "entities": greedy_prob(pe, "type", ENTITY_TYPES, thr)})
    return out


v1 = "xb xs wb ws db lb".split()
S_v1 = per_doc_stats(preds_for(v1), refs)
tests = [
    ("v3 all-8 vs v1 (LB said -0.0055)", "xb xs wb ws db ds lb ls"),
    ("v2 members (no XLM-R-large) vs v1 (LB -0.008 w/ priors)", "wb ws db ds lb ls"),
    ("v1 + mDeBERTa span vs v1 (saturation)", "xb xs wb ws db lb ds"),
    ("selection-biased best 5 vs v1", "wb ws db ds xs"),
    ("IndoBERTweet pair only vs v1", "wb ws"),
]
print("paired bootstrap over the 663-doc tail holdout, B=2000, threshold 0.3544")
for tag, names in tests:
    r = paired(S_v1, per_doc_stats(preds_for(names.split()), refs))
    print(fmt(tag, r), flush=True)
