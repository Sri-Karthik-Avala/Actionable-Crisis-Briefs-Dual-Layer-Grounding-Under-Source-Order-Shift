import sys
sys.path.insert(0,"dev")
from mix import *
bios=["xlmrl_crf0_last.npy","indobertweet_crf0_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy"]
ctrl=["span_ibtw_last.npy","m_mdeb_span_last.npy","m_ibtl_span_last.npy"]
wl=["v3_span_wl1_last.npy","v3_mdeb_span_wl1_last.npy","v3_ibtl_span_wl1_last.npy"]
cache={}
def get(n):
    if n not in cache: cache[n]=member_spans(n)
    return cache[n]
keys=None
for tag,names in [("bio4 + control spans",bios+ctrl),("bio4 + weighted spans",bios+wl)]:
    mem=[get(n) for n in names]
    if keys is None: keys=sorted(mem[0].keys()); refs=refs_of(keys)
    PA={k:np.mean([m[k][0] for m in mem],0) for k in keys}
    PE={k:np.mean([m[k][1] for m in mem],0) for k in keys}
    for thr in [0.3,0.3544,0.4]:
        s=score_docs([{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,thr),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,thr)} for k in keys],refs)
        print(f"{tag:24s} thr {thr:.4f}  {s[0]:.4f} (arg {s[1]:.4f} ent {s[2]:.4f})",flush=True)
