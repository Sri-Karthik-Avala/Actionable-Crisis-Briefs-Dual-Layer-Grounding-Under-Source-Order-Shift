import sys
sys.path.insert(0,"dev")
from mix import *
def span_f1(pred_docs, gold_docs, key, labels):
    sc=[]
    for lab in labels:
        tp=npd=ng=0
        for p,g in zip(pred_docs,gold_docs):
            ps={(x["start"],x["end"]) for x in p if x[key]==lab}; gs={(x["start"],x["end"]) for x in g if x[key]==lab}
            tp+=len(ps&gs); npd+=len(ps); ng+=len(gs)
        sc.append(2*tp/(npd+ng) if npd+ng else 1.0)
    return float(np.mean(sc))
for names in [["indobertweet_crf0_last.npy","span_ibtw_last.npy"], ["xlmrl_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy","indobertweet_crf0_last.npy","span_xlmrl_last.npy"]]:
    mem={n:member_spans(n) for n in names}; keys=sorted(mem[names[0]].keys()); refs=refs_of(keys)
    PA={k:np.mean([mem[n][k][0] for n in names],0) for k in keys}; PE={k:np.mean([mem[n][k][1] for n in names],0) for k in keys}
    for thr in [0.25,0.3,0.35,0.4,0.45,0.5,0.55]:
        pa=[greedy_prob(PA[k],"role",ARGUMENT_ROLES,thr) for k in keys]; pe=[greedy_prob(PE[k],"type",ENTITY_TYPES,thr) for k in keys]
        f=0.5*span_f1(pa,[r["arguments"] for r in refs],"role",ARGUMENT_ROLES)+0.5*span_f1(pe,[r["entities"] for r in refs],"type",ENTITY_TYPES)
        s=score_docs([{"arguments":a,"entities":e} for a,e in zip(pa,pe)],refs)
        print(len(names), thr, "macroF1 %.4f  metric %.4f"%(f,s[0]), flush=True)
