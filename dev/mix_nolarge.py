import sys
sys.path.insert(0,"dev")
from mix import *
sets={
 "v1 6 (with xlmr-large)":["xlmrl_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy","indobertweet_crf0_last.npy","span_xlmrl_last.npy"],
 "no-xlmrl 6":["indobertweet_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","m_mdeb_span_last.npy","indobertl_crf0_last.npy","m_ibtl_span_last.npy"],
 "all 8":["xlmrl_crf0_last.npy","span_xlmrl_last.npy","indobertweet_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","m_mdeb_span_last.npy","indobertl_crf0_last.npy","m_ibtl_span_last.npy"],
}
cache={}
def get(n):
    if n not in cache: cache[n]=member_spans(n)
    return cache[n]
keys=None
for tag,names in sets.items():
    mem=[get(n) for n in names]
    if keys is None: keys=sorted(mem[0].keys()); refs=refs_of(keys)
    PA={k:np.mean([m[k][0] for m in mem],0) for k in keys}
    PE={k:np.mean([m[k][1] for m in mem],0) for k in keys}
    best=max((score_docs([{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,t),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,t)} for k in keys],refs)[0],t) for t in [0.3,0.3544,0.4])
    print(tag,"best %.4f at thr %.4f"%best,flush=True)
