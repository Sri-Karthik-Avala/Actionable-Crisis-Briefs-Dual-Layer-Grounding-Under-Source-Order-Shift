import sys, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,z in BLOCKS:
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a)))
fit=[i for i in range(len(tr)) if chunk[i]<4]
first=collections.Counter(); last=collections.Counter(); inside=collections.Counter(); alltok=collections.Counter()
cover=collections.Counter()
for i in fit:
    w=tr.text[i].split(" "); t=T[tr.id[i]]
    covered=set()
    for x in t["arguments"]:
        first[w[x["start"]]]+=1; last[w[x["end"]-1]]+=1
        covered|=set(range(x["start"],x["end"]))
    for j,tok in enumerate(w):
        alltok[tok]+=1
        if j in covered: cover[tok]+=1
print("=== tokens that are FREQUENT but almost never START an argument span")
for tok,c in alltok.most_common(60):
    if c>=80:
        r=first[tok]/c
        if r<0.02: print(f"  {tok:12s} occurs {c:5d} starts {first[tok]:3d} ({r:.3f}) covered-anywhere {cover[tok]/c:.2f}")
print("\n=== tokens that are FREQUENT but almost never END an argument span")
for tok,c in alltok.most_common(60):
    if c>=80:
        r=last[tok]/c
        if r<0.02: print(f"  {tok:12s} occurs {c:5d} ends {last[tok]:3d} ({r:.3f})")
# how many inexact predictions are fixable by 1-token trim/extend?
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
preds=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
fixable=collections.Counter(); tot=collections.Counter()
lead=collections.Counter(); trail=collections.Counter()
for k,p,r in zip(keys,preds,refs):
    w=tr.text[k].split(" ")
    G={(x["role"],x["start"],x["end"]) for x in r["arguments"]}
    Gl=[(x["role"],x["start"],x["end"]) for x in r["arguments"]]
    for x in p["arguments"]:
        key=(x["role"],x["start"],x["end"])
        if key in G: tot["exact"]+=1; continue
        best=None
        for gl,gs,ge in Gl:
            if gl!=x["role"]: continue
            ov=max(0,min(ge,x["end"])-max(gs,x["start"]))
            if ov>0 and (best is None or ov>best[0]): best=(ov,gs,ge)
        if best is None: tot["nogold"]+=1; continue
        tot["inexact"]+=1
        _,gs,ge=best
        ds,de=x["start"]-gs, x["end"]-ge
        if abs(ds)<=1 and abs(de)<=1: fixable["within1"]+=1
        if ds==1 and de==0: lead["pred_drops_"+w[gs]]+=1
        if ds==0 and de==-1: trail["pred_drops_"+w[ge-1]]+=1
        if ds==-1 and de==0: lead["pred_adds_"+w[x["start"]]]+=1
        if ds==0 and de==1: trail["pred_adds_"+w[x["end"]-1]]+=1
print("\n=== argument predictions:",dict(tot))
print("inexact fixable by <=1 token at both ends:",fixable["within1"])
print("leading-token cases:",lead.most_common(10))
print("trailing-token cases:",trail.most_common(10))
