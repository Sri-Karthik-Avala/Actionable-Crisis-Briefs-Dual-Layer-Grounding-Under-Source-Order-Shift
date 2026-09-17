import sys, collections
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
tr=pd.read_csv("train.csv"); T=load_targets("train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,z in BLOCKS:
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a)))
fit=[i for i in range(len(tr)) if chunk[i]<4]
rateA={r:sum(len([x for x in T[tr.id[i]]["arguments"] if x["role"]==r]) for i in fit)/len(fit) for r in ARGUMENT_ROLES}
rateE={t:sum(len([x for x in T[tr.id[i]]["entities"] if x["type"]==t]) for i in fit)/len(fit) for t in ENTITY_TYPES}
bios=["xlmrl_crf0_last.npy","indobertweet_crf0_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy"]
ctrl=["span_ibtw_last.npy","m_mdeb_span_last.npy","m_ibtl_span_last.npy"]
wl=["v3_span_wl1_last.npy","v3_mdeb_span_wl1_last.npy","v3_ibtl_span_wl1_last.npy"]
cache={}
def get(n):
    if n not in cache: cache[n]=member_spans(n)
    return cache[n]
keys=None
def cnt(P,j,thr):
    c=0
    for k in keys:
        Q=P[k][:,:,1:]; lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]; sel=(lab==j)&(pr>thr)
        for s in range(m): c+=int(sel[s,:min(LS,m-s)].sum())
    return c
def thrs(P,labels,rate):
    out=[]
    for j,lab in enumerate(labels):
        target=rate[lab]*len(keys); lo,hi=0.05,0.9
        for _ in range(11):
            mid=(lo+hi)/2
            if cnt(P,j,mid)>target: lo=mid
            else: hi=mid
        out.append(min(max((lo+hi)/2,0.15),0.7))
    return np.array(out)
def dec(P,key,labels,ths):
    out=[]
    for k in keys:
        Q=P[k][:,:,1:]; lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]
        cand=[(pr[s,l]-ths[lab[s,l]],s,l,lab[s,l]) for s in range(m) for l in range(min(LS,m-s)) if pr[s,l]>ths[lab[s,l]]]
        cand.sort(key=lambda x:-x[0]); used=np.zeros(m,bool); sp=[]
        for _,s,l,c in cand:
            if not used[s:s+l+1].any(): used[s:s+l+1]=True; sp.append({key:labels[c],"start":int(s),"end":int(s+l+1)})
        out.append(sorted(sp,key=lambda x:x["start"]))
    return out
for tag,names in [("control spans",bios+ctrl),("all8",bios+ctrl+["span_xlmrl_last.npy"])]:
    mem=[get(n) for n in names]
    if keys is None: keys=sorted(mem[0].keys()); refs=refs_of(keys)
    PA={k:np.mean([m[k][0] for m in mem],0) for k in keys}
    PE={k:np.mean([m[k][1] for m in mem],0) for k in keys}
    ta=thrs(PA,ARGUMENT_ROLES,rateA); te=thrs(PE,ENTITY_TYPES,rateE)
    p=[{"arguments":a,"entities":e} for a,e in zip(dec(PA,"role",ARGUMENT_ROLES,ta),dec(PE,"type",ENTITY_TYPES,te))]
    print(f"{tag:16s} prior-matched  %.4f (arg %.4f ent %.4f)"%score_docs(p,refs),flush=True)
