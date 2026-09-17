import sys, pickle
sys.path.insert(0,"dev")
from mix import *
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
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
ta=np.full(len(ARGUMENT_ROLES),0.3544); te=np.full(len(ENTITY_TYPES),0.3544)
grid=[0.15,0.2,0.25,0.3,0.3544,0.4,0.45,0.5,0.6,0.7]
def sc(ta,te):
    return score_docs([{"arguments":a,"entities":e} for a,e in zip(dec(PA,"role",ARGUMENT_ROLES,ta),dec(PE,"type",ENTITY_TYPES,te))],refs)[0]
best=sc(ta,te); print("start %.4f"%best,flush=True)
for it in range(2):
    for j in range(len(ARGUMENT_ROLES)):
        for g in grid:
            t2=ta.copy(); t2[j]=g; v=sc(t2,te)
            if v>best+1e-9: best=v; ta=t2
    for j in range(len(ENTITY_TYPES)):
        for g in grid:
            t2=te.copy(); t2[j]=g; v=sc(ta,t2)
            if v>best+1e-9: best=v; te=t2
    print("iter",it,"%.4f"%best,flush=True)
print("oracle per-label thresholds arg:",[round(float(x),3) for x in ta])
print("oracle ent:",[round(float(x),3) for x in te])
print("ORACLE CEILING %.4f (baseline 0.7116)"%best)
