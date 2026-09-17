import sys, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
tr=pd.read_csv("train.csv")
keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
preds=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,0.3544),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,0.3544)} for k in keys]
for layer,fld in (("arguments","role"),("entities","type")):
    fp=collections.Counter(); miss=collections.Counter()
    for k,p,r in zip(keys,preds,refs):
        G=[(x[fld],x["start"],x["end"]) for x in r[layer]]
        P=[(x[fld],x["start"],x["end"]) for x in p[layer]]
        for pl,ps,pe in P:
            same=[g for g in G if g[0]==pl and max(0,min(g[2],pe)-max(g[1],ps))>0]
            if same: fp["matched_same_label"]+=1; continue
            other=[g for g in G if g[0]!=pl and max(0,min(g[2],pe)-max(g[1],ps))>0]
            if other:
                fp["wrong_label"]+=1; fp["conf:"+pl+"->"+other[0][0]]+=1
            else: fp["no_gold_span_here"]+=1
        for gl,gs,ge in G:
            same=[q for q in P if q[0]==gl and max(0,min(q[2],ge)-max(q[1],gs))>0]
            if same: miss["found"]+=1; continue
            other=[q for q in P if q[0]!=gl and max(0,min(q[2],ge)-max(q[1],gs))>0]
            miss["wrong_label" if other else "nothing_predicted"]+=1
    print(f"=== {layer}")
    print("  predictions:",{k:v for k,v in fp.items() if not k.startswith("conf:")})
    print("  gold:",dict(miss))
    print("  top confusions:",[(k[5:],v) for k,v in fp.most_common(60) if k.startswith("conf:")][:8])
