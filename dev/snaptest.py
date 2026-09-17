import sys, os
sys.path.insert(0,"dev")
from mix import *
O="colab_out/out/"
def local(name):
    d=np.load("dev/out/"+name,allow_pickle=True).item(); keys=sorted(d.keys()); out={}
    for k in keys:
        a,e=d[k]
        if a.ndim==3:
            pa=np.zeros((a.shape[0],LS,a.shape[2])); pa[:,:a.shape[1]]=np.exp(a.astype(np.float64))
            pe=np.zeros((e.shape[0],LS,e.shape[2])); pe[:,:e.shape[1]]=np.exp(e.astype(np.float64))
            out[k]=(pa,pe)
        else:
            out[k]=(bio_to_span(a,ARG_TAGS,ARGUMENT_ROLES), bio_to_span(e,ENT_TAGS,ENTITY_TYPES))
    return out
base=sys.argv[1]; eps=[int(x) for x in sys.argv[2].split(",")]
mems={e:local(f"{base}_ep{e}.npy") for e in eps}
keys=sorted(mems[eps[0]].keys()); refs=refs_of(keys)
for use in [[eps[-1]], eps[-2:], eps[-3:], eps]:
    PA={k:np.mean([mems[e][k][0] for e in use],0) for k in keys}
    PE={k:np.mean([mems[e][k][1] for e in use],0) for k in keys}
    best=max((score_docs([{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,t),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,t)} for k in keys],refs)[0],t) for t in [0.3,0.3544,0.4,0.45])
    print(base,"snapshots",use,"best %.4f at thr %s"%best,flush=True)
