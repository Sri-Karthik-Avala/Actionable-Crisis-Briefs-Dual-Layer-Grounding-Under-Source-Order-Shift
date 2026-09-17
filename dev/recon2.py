import pandas as pd, numpy as np, collections, sys, random
sys.path.insert(0,"dev")
from abgs_metric import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); te=pd.read_csv(D+"test.csv"); T=load_targets(D+"train_targets.jsonl")
for ph in ["<USER>","<URL>","<EMAIL>","<PHONE>","<EMPTY>"]:
    print(ph, sum(t.split(" ").count(ph) for t in tr.text), sum(t.split(" ").count(ph) for t in te.text))
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,b in BLOCKS:
    for i in range(b-a): chunk[a+i]=min(4,i*5//(b-a))
random.seed(0)
ex=collections.defaultdict(list)
seen_before=collections.Counter(); tot_val=collections.Counter()
early=collections.defaultdict(set)
for i,r in tr.iterrows():
    w=r.text.split(" "); t=T[r.id]
    for key,lab in [("arguments","role"),("entities","type")]:
        for s in t[key]:
            txt=" ".join(w[s["start"]:s["end"]])
            if len(ex[s[lab]])<400: ex[s[lab]].append(txt)
            if chunk[i]<4: early[s[lab]].add(txt)
for i,r in tr.iterrows():
    if chunk[i]!=4: continue
    w=r.text.split(" "); t=T[r.id]
    for key,lab in [("arguments","role"),("entities","type")]:
        for s in t[key]:
            txt=" ".join(w[s["start"]:s["end"]]); tot_val[s[lab]]+=1; seen_before[s[lab]]+=txt in early[s[lab]]
for k in ARGUMENT_ROLES+ENTITY_TYPES:
    c=collections.Counter(ex[k]).most_common(8)
    print(f"{k:20s} val-seen {seen_before[k]}/{tot_val[k]}  top:",c)
    print("     rand:", random.sample(ex[k], min(6,len(ex[k]))))
# a few full examples with victims
k=0
for i,r in tr.iterrows():
    t=T[r.id]
    if any(a["role"] in ("WOUNDVICTIM-ARG","DEATHVICTIM-ARG","REASON-ARG") for a in t["arguments"]):
        w=r.text.split(" ")
        print("\n",r.text)
        print("  A:",[(a["role"]," ".join(w[a["start"]:a["end"]])) for a in t["arguments"]])
        print("  E:",[(a["type"]," ".join(w[a["start"]:a["end"]])) for a in t["entities"]])
        k+=1
        if k>=6: break
