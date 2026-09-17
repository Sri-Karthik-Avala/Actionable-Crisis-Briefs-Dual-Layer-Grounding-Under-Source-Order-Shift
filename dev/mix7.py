import sys, pickle
sys.path.insert(0,"dev")
from mix import *
base=["xlmrl_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy","indobertweet_crf0_last.npy","span_xlmrl_last.npy"]
new=["m_mdeb_span_last.npy"]
cache={n:member_spans(n) for n in base+new}
keys=sorted(cache[base[0]].keys()); refs=refs_of(keys)
for names,tag in [(base,"6 (v1)"),(base+new,"7 (+mdeb span)")]:
    PA={k:np.mean([cache[n][k][0] for n in names],0) for k in keys}
    PE={k:np.mean([cache[n][k][1] for n in names],0) for k in keys}
    best=max((score_docs([{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,t),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,t)} for k in keys],refs)[0],t) for t in [0.3,0.3544,0.4])
    print(tag,"best %.4f at thr %.4f"%best,flush=True)
    if len(names)==7: pickle.dump((keys,PA,PE),open("dev/out_mix7.pkl","wb"))
