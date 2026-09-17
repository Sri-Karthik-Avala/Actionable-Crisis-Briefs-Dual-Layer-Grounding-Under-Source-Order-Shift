import sys, pickle, os
sys.path.insert(0,"dev")
from mix import *
names=["xlmrl_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy","indobertweet_crf0_last.npy","span_xlmrl_last.npy"]
cp="dev/out_mix6.pkl"
if os.path.exists(cp):
    keys,PA,PE=pickle.load(open(cp,"rb"))
else:
    mem={n:member_spans(n) for n in names}; keys=sorted(mem[names[0]].keys())
    PA={k:np.mean([mem[n][k][0] for n in names],0) for k in keys}; PE={k:np.mean([mem[n][k][1] for n in names],0) for k in keys}
    pickle.dump((keys,PA,PE),open(cp,"wb"))
refs=refs_of(keys)
def dec(ks, ta, te, ma=None, me=None):
    return [{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,ta,ma),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,te,me)} for k in ks]
print("layer thresholds")
for ta in [0.3,0.35,0.4]:
    for te in [0.3,0.35,0.4,0.45,0.5]:
        print(ta,te,"%.4f"%score_docs(dec(keys,ta,te),refs)[0],flush=True)
rng=np.random.RandomState(0); perm=rng.permutation(len(keys))
halves=[sorted(perm[:len(perm)//2]),sorted(perm[len(perm)//2:])]
grid=[0.6,0.8,1.0,1.25,1.5]
gains=[]
for a,b in [(0,1),(1,0)]:
    ka=[keys[i] for i in halves[a]]; kb=[keys[i] for i in halves[b]]
    ra=[refs[i] for i in halves[a]]; rb=[refs[i] for i in halves[b]]
    ma=np.ones(len(ARGUMENT_ROLES)); me=np.ones(len(ENTITY_TYPES))
    for j in range(len(ARGUMENT_ROLES)):
        best=None
        for g in grid:
            m=ma.copy(); m[j]=g; s=score_docs(dec(ka,0.35,0.35,m,me),ra)[0]
            if best is None or s>best[0]+1e-9: best=(s,g)
        ma[j]=best[1]
    for j in range(len(ENTITY_TYPES)):
        best=None
        for g in grid:
            m=me.copy(); m[j]=g; s=score_docs(dec(ka,0.35,0.35,ma,m),ra)[0]
            if best is None or s>best[0]+1e-9: best=(s,g)
        me[j]=best[1]
    s0=score_docs(dec(kb,0.35,0.35),rb)[0]; s1=score_docs(dec(kb,0.35,0.35,ma,me),rb)[0]
    gains.append(s1-s0); print("mult",dict(zip(ARGUMENT_ROLES,ma)),dict(zip(ENTITY_TYPES,me)),"%.4f -> %.4f"%(s0,s1),flush=True)
print("crossfit per-label gain",np.mean(gains))
