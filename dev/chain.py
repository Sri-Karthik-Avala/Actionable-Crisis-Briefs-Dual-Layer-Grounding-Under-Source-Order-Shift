import sys, collections
import pandas as pd, numpy as np
sys.path.insert(0,"dev")
from abgs_metric import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
pat=collections.Counter(); first=collections.Counter(); chains=0
pos_label=collections.Counter()
for i,r in tr.iterrows():
    ents=[e for e in T[r.id]["entities"] if e["type"] in ("LOC","PLOC")]
    ents.sort(key=lambda x:x["start"])
    cur=[]
    for e in ents:
        if cur and e["start"]-cur[-1]["end"]<=1: cur.append(e)
        else:
            if cur: 
                chains+=1; pat["".join("L" if x["type"]=="LOC" else "P" for x in cur)]+=1
                for k,x in enumerate(cur): pos_label[(min(k,3),x["type"])]+=1
            cur=[e]
    if cur:
        chains+=1; pat["".join("L" if x["type"]=="LOC" else "P" for x in cur)]+=1
        for k,x in enumerate(cur): pos_label[(min(k,3),x["type"])]+=1
print("chains",chains)
print("top patterns",pat.most_common(12))
multi=sum(v for k,v in pat.items() if len(k)>1)
print("multi-span chains",multi)
for k in range(4):
    l=pos_label[(k,"LOC")]; p=pos_label[(k,"PLOC")]
    print(f"position {k}: LOC {l} PLOC {p}  -> P(LOC)={l/(l+p+1e-9):.2f}")
lp=sum(v for k,v in pat.items() if len(k)>1 and k[0]=="L" and set(k[1:])<= {"P"})
print("chains matching L then all-P:",lp,"of",multi)
