import pandas as pd, numpy as np, collections, sys
sys.path.insert(0,"dev")
from abgs_metric import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
EV=["FIRE-EVENT","FLOOD-EVENT","EARTHQUAKE-EVENT","ACCIDENT-EVENT","FALSE-EVENT"]
multi=0; mixed=collections.Counter(); blk_ev=collections.defaultdict(collections.Counter)
for i,r in tr.iterrows():
    b=[k for k,(a,z) in enumerate(BLOCKS) if a<=i<z][0]
    ev=[a["role"] for a in T[r.id]["arguments"] if a["role"] in EV]
    for e in ev: blk_ev[b][e]+=1
    if len(ev)>1:
        multi+=1
        if len(set(ev))>1: mixed[tuple(sorted(set(ev)))]+=1
print("docs with >1 event span",multi,"mixed role docs",sum(mixed.values()),mixed.most_common(8))
for b in range(4): print("block",b,dict(blk_ev[b]))
w_role=collections.defaultdict(collections.Counter)
for i,r in tr.iterrows():
    w=r.text.split(" ")
    for a in T[r.id]["arguments"]:
        if a["role"] in EV: w_role[" ".join(w[a["start"]:a["end"]])][a["role"]]+=1
print("trigger text -> roles", {k:dict(v) for k,v in sorted(w_role.items(), key=lambda x:-sum(x[1].values()))[:15]})
unl=0; lab=0
for i,r in tr.iterrows():
    w=r.text.split(" "); cov=set()
    for a in T[r.id]["arguments"]:
        cov|=set(range(a["start"],a["end"]))
    for j,x in enumerate(w):
        if x in ("banjir","kebakaran","gempa","kecelakaan"):
            if j in cov: lab+=1
            else: unl+=1
print("keyword tokens covered by an arg span",lab,"uncovered",unl)
