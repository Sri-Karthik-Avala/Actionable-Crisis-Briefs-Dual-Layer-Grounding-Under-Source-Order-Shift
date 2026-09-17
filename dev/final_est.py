import sys, pickle
sys.path.insert(0,"dev")
from mix import *
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb"))
refs=refs_of(keys)
for thr in [0.3544]:
    preds=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,thr),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,thr)} for k in keys]
    print("threshold",thr)
    score_docs(preds,refs,verbose=True)
