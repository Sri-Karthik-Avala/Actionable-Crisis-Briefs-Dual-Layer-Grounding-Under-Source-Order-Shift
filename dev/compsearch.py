import sys
sys.path.insert(0,"dev")
from mix import *
P={"xb":"xlmrl_crf0_last.npy","xs":"span_xlmrl_last.npy","wb":"indobertweet_crf0_last.npy","ws":"span_ibtw_last.npy",
   "db":"mdeb_crf0_last.npy","ds":"m_mdeb_span_last.npy","lb":"indobertl_crf0_last.npy","ls":"m_ibtl_span_last.npy"}
cache={}
def get(n):
    if n not in cache: cache[n]=member_spans(P[n])
    return cache[n]
combos=[("wb ws",),("db ds",),("lb ls",),("wb ws db ds",),("wb ws lb ls",),("db ds lb ls",),
        ("wb ws db ds lb ls",),("wb ws db ds lb ls xb",),("wb ws db ds lb ls xs",),("wb ws db ds lb ls xb xs",),
        ("wb db lb",),("ws ds ls",),("wb ws db ds xs",)]
keys=None
res=[]
for (c,) in combos:
    names=c.split()
    mem=[get(n) for n in names]
    if keys is None: keys=sorted(mem[0].keys()); refs=refs_of(keys)
    PA={k:np.mean([m[k][0] for m in mem],0) for k in keys}
    PE={k:np.mean([m[k][1] for m in mem],0) for k in keys}
    s=score_docs([{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys],refs)
    res.append((s[0],c)); print("%-26s %.4f (arg %.4f ent %.4f)"%(c,s[0],s[1],s[2]),flush=True)
print("\nBEST:"); [print("%.4f %s"%x) for x in sorted(res)[-4:]]
