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
rateA={r:0 for r in ARGUMENT_ROLES}; rateE={t:0 for t in ENTITY_TYPES}
for i in fit:
    t=T[tr.id[i]]
    for a in t["arguments"]: rateA[a["role"]]+=1
    for e in t["entities"]: rateE[e["type"]]+=1
for k in rateA: rateA[k]/=len(fit)
for k in rateE: rateE[k]/=len(fit)
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
n=len(keys)
def counts_at(P,j,thr):
    # count spans whose argmax label is j and prob>thr (upper bound, ignores overlap resolution)
    c=0
    for k in keys:
        Q=P[k][:,:,1:]; lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]
        for s in range(m):
            for l in range(min(LS,m-s)):
                if lab[s,l]==j and pr[s,l]>thr: c+=1
    return c
def per_label_thr(P,labels,rate,base):
    ths=[]
    for j,lab in enumerate(labels):
        target=rate[lab]*n
        lo,hi=0.05,0.9
        for _ in range(12):
            mid=(lo+hi)/2
            c=counts_at(P,j,mid)
            if c>target: lo=mid
            else: hi=mid
        ths.append(min(max((lo+hi)/2,0.15),0.7))
    return np.array(ths)
def dec_pl(P,key,labels,ths):
    out=[]
    for k in keys:
        Q=P[k][:,:,1:].copy()
        lab=Q.argmax(-1); pr=np.take_along_axis(Q,lab[:,:,None],-1)[:,:,0]
        m=P[k].shape[0]
        cand=[(pr[s,l]-ths[lab[s,l]],s,l,lab[s,l]) for s in range(m) for l in range(min(LS,m-s)) if pr[s,l]>ths[lab[s,l]]]
        cand.sort(key=lambda x:-x[0]); used=np.zeros(m,bool); sp=[]
        for _,s,l,c in cand:
            if not used[s:s+l+1].any():
                used[s:s+l+1]=True; sp.append({key:labels[c],"start":int(s),"end":int(s+l+1)})
        out.append(sorted(sp,key=lambda x:x["start"]))
    return out
ta=per_label_thr(PA,ARGUMENT_ROLES,rateA,0.3544); te=per_label_thr(PE,ENTITY_TYPES,rateE,0.3544)
print("arg thr:",{r:round(float(x),3) for r,x in zip(ARGUMENT_ROLES,ta)})
print("ent thr:",{t:round(float(x),3) for t,x in zip(ENTITY_TYPES,te)})
pa=dec_pl(PA,"role",ARGUMENT_ROLES,ta); pe=dec_pl(PE,"type",ENTITY_TYPES,te)
print("prior-matched %.4f %.4f %.4f"%score_docs([{"arguments":a,"entities":e} for a,e in zip(pa,pe)],refs))
base=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
print("baseline       %.4f %.4f %.4f"%score_docs(base,refs))
# blend: shrink per-label thresholds toward global
for w in [0.25,0.5,0.75]:
    ta2=w*ta+(1-w)*0.3544; te2=w*te+(1-w)*0.3544
    pa=dec_pl(PA,"role",ARGUMENT_ROLES,ta2); pe=dec_pl(PE,"type",ENTITY_TYPES,te2)
    print("shrink %.2f     %.4f %.4f %.4f"%((w,)+score_docs([{"arguments":a,"entities":e} for a,e in zip(pa,pe)],refs)),flush=True)
