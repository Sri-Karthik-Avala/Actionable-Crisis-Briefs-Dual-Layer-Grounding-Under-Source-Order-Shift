import sys, pickle
import pandas as pd
sys.path.insert(0,"dev")
from mix import *
D="C:/Users/srika/Downloads/eris_actionable/"
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
chunk=np.zeros(len(tr),int)
for a,b in BLOCKS:
    for i in range(b-a): chunk[a+i]=min(4,int(i*5//(b-a)))
fit=[i for i in range(len(tr)) if chunk[i]!=4]
rateA={r:0.0 for r in ARGUMENT_ROLES}; rateE={t:0.0 for t in ENTITY_TYPES}
for i in fit:
    t=T[tr.id[i]]
    for a in t["arguments"]: rateA[a["role"]]+=1
    for e in t["entities"]: rateE[e["type"]]+=1
for k in rateA: rateA[k]/=len(fit)
for k in rateE: rateE[k]/=len(fit)
keys,PA6,PE6=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
names2=["indobertweet_crf0_last.npy","span_ibtw_last.npy"]
mem={n:member_spans(n) for n in names2}
PA2={k:np.mean([mem[n][k][0] for n in names2],0) for k in keys}
PE2={k:np.mean([mem[n][k][1] for n in names2],0) for k in keys}
n=len(keys)
def cnt(P,j,thr):
    c=0
    for k in keys:
        Q=P[k][:,:,1:]; lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]
        sel=(lab==j)&(pr>thr)
        for s in range(m):
            c+=int(sel[s,:min(LS,m-s)].sum())
    return c
def thrs(P,labels,rate):
    out=[]
    for j,lab in enumerate(labels):
        target=rate[lab]*n; lo,hi=0.05,0.9
        for _ in range(12):
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
ta2=thrs(PA2,ARGUMENT_ROLES,rateA); te2=thrs(PE2,ENTITY_TYPES,rateE)
ta6=thrs(PA6,ARGUMENT_ROLES,rateA); te6=thrs(PE6,ENTITY_TYPES,rateE)
print("thr from 2-member:",[round(float(x),3) for x in ta2])
print("thr from 6-member:",[round(float(x),3) for x in ta6])
for name,(ta,te) in [("self (6-member thr)",(ta6,te6)),("transferred (2-member thr)",(ta2,te2))]:
    p=[{"arguments":a,"entities":e} for a,e in zip(dec(PA6,"role",ARGUMENT_ROLES,ta),dec(PE6,"type",ENTITY_TYPES,te))]
    print(name,"%.4f %.4f %.4f"%score_docs(p,refs),flush=True)
base=[{"arguments":greedy_prob(PA6[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE6[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
print("global thr baseline %.4f"%score_docs(base,refs)[0])
