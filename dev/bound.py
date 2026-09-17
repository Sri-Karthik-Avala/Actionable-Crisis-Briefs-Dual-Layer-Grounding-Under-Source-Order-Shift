import sys, pickle, collections
sys.path.insert(0,"dev")
from mix import *
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
preds=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
ds=collections.Counter(); de=collections.Counter(); per=collections.defaultdict(lambda: [0,0,0])
tr_df=pd.read_csv("train.csv")
for key,p,r in zip(keys,preds,refs):
    for layer,f in (("arguments","role"),("entities","type")):
        G=[(x[f],x["start"],x["end"]) for x in r[layer]]
        for x in p[layer]:
            best=None
            for gl,gs,ge in G:
                if gl!=x[f]: continue
                ov=max(0,min(ge,x["end"])-max(gs,x["start"]))
                if ov>0 and (best is None or ov>best[0]): best=(ov,gs,ge)
            if best is None: continue
            _,gs,ge=best
            if (gs,ge)==(x["start"],x["end"]): per[x[f]][0]+=1; continue
            per[x[f]][1]+=1
            ds[x["start"]-gs]+=1; de[x["end"]-ge]+=1
print("start delta (pred-gold):",dict(sorted(ds.items())))
print("end   delta (pred-gold):",dict(sorted(de.items())))
tot_s=sum(ds.values()); print("share start-correct among inexact: %.2f"%(ds[0]/tot_s), " end-correct: %.2f"%(de[0]/sum(de.values())))
print("exact/inexact per label:", {k:(v[0],v[1]) for k,v in sorted(per.items())})
