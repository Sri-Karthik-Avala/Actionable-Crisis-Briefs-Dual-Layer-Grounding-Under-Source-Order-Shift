import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,z in BLOCKS:
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a)))
names=["xlmrl_crf0_last.npy","span_xlmrl_last.npy","indobertweet_crf0_last.npy","span_ibtw_last.npy","mdeb_crf0_last.npy","indobertl_crf0_last.npy"]
mem=[member_spans(n) for n in names]; keys=sorted(mem[0].keys()); refs=refs_of(keys)
PA={k:np.mean([m[k][0] for m in mem],0) for k in keys}; PE={k:np.mean([m[k][1] for m in mem],0) for k in keys}
hold_len=np.array([tr.n_tokens[k] for k in keys])
def counts_fit(idx,layer,fld,lab):
    return np.array([len([x for x in T[tr.id[i]][layer] if x[fld]==lab]) for i in idx]), np.array([tr.n_tokens[i] for i in idx])
def predict_total(idx_fit,layer,fld,lab,lens_target):
    y,L=counts_fit(idx_fit,layer,fld,lab)
    A=np.stack([np.ones_like(L,dtype=float),L.astype(float)],1)
    coef,*_=np.linalg.lstsq(A,y.astype(float),rcond=None)
    pred=coef[0]+coef[1]*lens_target
    return float(np.clip(pred,0,None).sum())
fit_all=[i for i in range(len(tr)) if chunk[i]<4]
fit_recent=[i for i in range(len(tr)) if chunk[i]==3]
def cnt(P,j,thr):
    c=0
    for k in keys:
        Q=P[k][:,:,1:]; lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]; sel=(lab==j)&(pr>thr)
        for s in range(m): c+=int(sel[s,:min(LS,m-s)].sum())
    return c
def thr_for(P,j,target):
    lo,hi=0.05,0.9
    for _ in range(11):
        mid=(lo+hi)/2
        if cnt(P,j,mid)>target: lo=mid
        else: hi=mid
    return min(max((lo+hi)/2,0.15),0.7)
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
base=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
print("global thr       %.4f"%score_docs(base,refs)[0])
for tag,fit_idx,mode in [("per-doc  all",fit_all,"doc"),("per-len  all",fit_all,"len"),("per-len  recent",fit_recent,"len")]:
    ta=[];te_=[]
    for j,lab in enumerate(ARGUMENT_ROLES):
        if mode=="doc":
            y,_=counts_fit(fit_idx,"arguments","role",lab); target=y.mean()*len(keys)
        else: target=predict_total(fit_idx,"arguments","role",lab,hold_len)
        ta.append(thr_for(PA,j,target))
    for j,lab in enumerate(ENTITY_TYPES):
        if mode=="doc":
            y,_=counts_fit(fit_idx,"entities","type",lab); target=y.mean()*len(keys)
        else: target=predict_total(fit_idx,"entities","type",lab,hold_len)
        te_.append(thr_for(PE,j,target))
    p=[{"arguments":a,"entities":e} for a,e in zip(dec(PA,"role",ARGUMENT_ROLES,np.array(ta)),dec(PE,"type",ENTITY_TYPES,np.array(te_)))]
    print(f"{tag:16s} %.4f"%score_docs(p,refs)[0],flush=True)
