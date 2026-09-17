import sys, numpy as np, collections
sys.path.insert(0,"dev")
from ens import *
res=load(sys.argv[1:]); keys=sorted(res.keys()); refs=refs_of(keys); preds=decode(res,keys)
base=score_docs(preds,refs)[0]
rng=np.random.RandomState(0); bs=[]
for b in range(200):
    ix=rng.randint(0,len(keys),len(keys))
    bs.append(score_docs([preds[i] for i in ix],[refs[i] for i in ix])[0])
print(f"score {base:.4f} bootstrap std {np.std(bs):.4f}")
def conf(layer,lab):
    c=collections.Counter()
    for p,r in zip(preds,refs):
        R=[(x[lab],x["start"],x["end"]) for x in r[layer]]
        for x in p[layer]:
            best=None
            for rl,rs,re_ in R:
                ov=max(0,min(re_,x["end"])-max(rs,x["start"]))
                if ov>0 and (best is None or ov>best[0]): best=(ov,rl,(rs,re_)==(x["start"],x["end"]))
            c[(x[lab], "NONE" if best is None else best[1]+("=" if best[2] else "~"))]+=1
    return c
c=conf("arguments","role")
for role in ["WOUNDVICTIM-ARG","DEATHVICTIM-ARG","STREET-ARG","INFORMATION-ARG","REASON-ARG","AFFECTEDOBJECTS-ARG","EARTHQUAKE-EVENT"]:
    print("pred",role,{k[1]:v for k,v in c.items() if k[0]==role})
# gold-side: what did we predict at gold spans
g=collections.Counter()
for p,r in zip(preds,refs):
    P=[(x["role"],x["start"],x["end"]) for x in p["arguments"]]
    for x in r["arguments"]:
        best=None
        for pl,ps,pe in P:
            ov=max(0,min(pe,x["end"])-max(ps,x["start"]))
            if ov>0 and (best is None or ov>best[0]): best=(ov,pl,(ps,pe)==(x["start"],x["end"]))
        g[(x["role"],"MISS" if best is None else best[1]+("=" if best[2] else "~"))]+=1
for role in ["WOUNDVICTIM-ARG","DEATHVICTIM-ARG","STREET-ARG","INFORMATION-ARG","EARTHQUAKE-EVENT"]:
    print("gold",role,{k[1]:v for k,v in g.items() if k[0]==role})
