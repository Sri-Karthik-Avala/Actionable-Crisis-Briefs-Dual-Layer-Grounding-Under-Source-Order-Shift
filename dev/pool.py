import sys, pickle, collections
sys.path.insert(0,"dev")
from mix import *
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
for thr in [0.02,0.05,0.1,0.2]:
    stat=collections.Counter(); tot=collections.Counter()
    for k,r in zip(keys,refs):
        for layer,f,P,labels in (("arguments","role",PA,ARGUMENT_ROLES),("entities","type",PE,ENTITY_TYPES)):
            Q=P[k]
            for x in r[layer]:
                j=labels.index(x[f]); s=x["start"]; l=x["end"]-x["start"]-1
                tot[layer]+=1
                if l<LS and Q[s,l,j+1]>thr: stat[layer]+=1
    print("thr %.2f  arg pool recall %.3f (%d/%d)  ent %.3f (%d/%d)"%(thr,stat["arguments"]/tot["arguments"],stat["arguments"],tot["arguments"],stat["entities"]/tot["entities"],stat["entities"],tot["entities"]),flush=True)
# rank of gold among candidates for missed labels
miss=collections.Counter(); ranks=collections.defaultdict(list)
for k,r in zip(keys,refs):
    Q=PA[k]
    for x in r["arguments"]:
        j=ARGUMENT_ROLES.index(x["role"]); s=x["start"]; l=x["end"]-x["start"]-1
        if l>=LS: continue
        p=Q[s,l,j+1]
        better=(Q[:,:,1:].max(-1)>p).sum()
        ranks[x["role"]].append(int(better))
for role in ["WOUNDVICTIM-ARG","STREET-ARG","INFORMATION-ARG","PLACE-ARG"]:
    rs=np.array(ranks[role]); print(role,"median rank of gold among all candidates: %.0f  frac rank0: %.2f  frac rank<5: %.2f"%(np.median(rs),(rs==0).mean(),(rs<5).mean()))
