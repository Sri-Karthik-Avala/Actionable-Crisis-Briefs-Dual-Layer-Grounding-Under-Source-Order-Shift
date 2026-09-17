import sys, pickle
sys.path.insert(0,"dev")
from mix import *
names=["xlmrl_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy","indobertweet_crf0_last.npy","span_xlmrl_last.npy"]
mem={n:member_spans(n) for n in names}; keys=sorted(mem[names[0]].keys())
PA={k:np.mean([mem[n][k][0] for n in names],0) for k in keys}; PE={k:np.mean([mem[n][k][1] for n in names],0) for k in keys}
pickle.dump((keys,PA,PE),open("dev/out_mix6.pkl","wb"))
refs=refs_of(keys)
for thr in [0.3,0.3544,0.4]:
    preds=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,thr),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,thr)} for k in keys]
    print("baseline thr",thr,"%.4f %.4f %.4f"%score_docs(preds,refs),flush=True)
