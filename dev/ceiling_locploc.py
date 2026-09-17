import sys, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
tr=pd.read_csv("train.csv"); T=load_targets("train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,z in BLOCKS:
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a)))
ref=collections.defaultdict(collections.Counter)
for i in range(len(tr)):
    if chunk[i]==4: continue
    w=tr.text[i].split(" ")
    for x in T[tr.id[i]]["entities"]:
        if x["type"] in ("LOC","PLOC"): ref[" ".join(w[x["start"]:x["end"]])][x["type"]]+=1
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb"))
refs=refs_of(keys)
preds=[{"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
seen=hit_major=hit_model=unseen=0
for k,p,r in zip(keys,preds,refs):
    w=tr.text[k].split(" ")
    pm={(x["start"],x["end"]):x["type"] for x in p["entities"]}
    for x in r["entities"]:
        if x["type"] not in ("LOC","PLOC"): continue
        s=" ".join(w[x["start"]:x["end"]])
        c=ref.get(s)
        if not c: unseen+=1; continue
        seen+=1
        major=c.most_common(1)[0][0]
        hit_major+= (major==x["type"])
        mt=pm.get((x["start"],x["end"]))
        hit_model+= (mt==x["type"])
print(f"holdout LOC/PLOC gold spans whose string was seen in train: {seen} (unseen {unseen})")
print(f"  majority-string oracle accuracy: {hit_major/seen:.3f}")
print(f"  our ensemble accuracy (exact span + type): {hit_model/seen:.3f}")
amb=[(s,c) for s,c in ref.items() if sum(c.values())>=4]
frac=np.mean([min(c.values())/sum(c.values()) for s,c in amb])
print(f"  mean minority share for strings seen >=4x: {frac:.3f}  (0.5 = pure coin flip)")
tot=sum(sum(c.values()) for c in ref.values())
best=sum(max(c.values()) for c in ref.values())
print(f"  train-side Bayes bound from string identity alone: {best/tot:.3f}")
