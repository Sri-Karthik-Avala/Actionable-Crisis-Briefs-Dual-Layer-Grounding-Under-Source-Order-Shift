import sys, collections
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from abgs_metric import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); T=load_targets(D+"train_targets.jsonl")
street_cue=["jl","jl.","jalan","km","gg","gang","rw","rt","tol","raya","jln"]
adm=["kecamatan","kabupaten","kota","desa","provinsi","kelurahan","kab","prov","dusun","kec"]
phys=["sungai","gunung","pasar","pantai","kali","jembatan","hutan","danau","bandara","terminal"]
sc=collections.Counter(); ac=collections.Counter()
for i,r in tr.iterrows():
    w=r.text.split(" "); t=T[r.id]
    for x in t["arguments"]:
        if x["role"] not in ("STREET-ARG","PLACE-ARG"): continue
        toks=[w[j] for j in range(x["start"],x["end"])]
        has=any(any(tok==c or tok.startswith(c+".") for c in street_cue) for tok in toks)
        sc[(x["role"],has)]+=1
    for x in t["entities"]:
        if x["type"] not in ("LOC","PLOC"): continue
        toks=[w[j] for j in range(x["start"],x["end"])]
        a=any(tok in adm for tok in toks); p=any(tok in phys for tok in toks)
        ac[(x["type"],"adm" if a else ("phys" if p else "bare"))]+=1
print("=== STREET vs PLACE by street-cue token (jl/jalan/km/gg/rw/tol/raya)")
for role in ("STREET-ARG","PLACE-ARG"):
    y,n=sc[(role,True)],sc[(role,False)]
    print(f"  {role:12s} cue {y:5d}  no-cue {n:5d}   P(cue)={y/(y+n):.2f}")
tot_cue=sc[("STREET-ARG",True)]+sc[("PLACE-ARG",True)]
print(f"  => given a street cue, P(STREET) = {sc[('STREET-ARG',True)]/max(1,tot_cue):.2f}")
tot_no=sc[("STREET-ARG",False)]+sc[("PLACE-ARG",False)]
print(f"  => given NO cue,      P(STREET) = {sc[('STREET-ARG',False)]/max(1,tot_no):.2f}")
print("\n=== LOC vs PLOC by head-word class")
for cls in ("adm","phys","bare"):
    l,p=ac[("LOC",cls)],ac[("PLOC",cls)]
    print(f"  {cls:5s} LOC {l:5d} PLOC {p:5d}   P(PLOC|{cls})={p/max(1,l+p):.2f}")
# per-string consistency: same span text labelled differently across docs?
txt=collections.defaultdict(collections.Counter)
for i,r in tr.iterrows():
    w=r.text.split(" ")
    for x in T[r.id]["entities"]:
        if x["type"] in ("LOC","PLOC"): txt[" ".join(w[x["start"]:x["end"]])][x["type"]]+=1
multi=[(k,v) for k,v in txt.items() if sum(v.values())>=5]
incons=[(k,v) for k,v in multi if min(v.values())/sum(v.values())>0.25]
print(f"\nstrings seen >=5 times: {len(multi)}; labelled inconsistently (minority>25%): {len(incons)}")
print("examples:",[(k,dict(v)) for k,v in incons[:10]])
print("\n=== same for STREET/PLACE strings")
txt2=collections.defaultdict(collections.Counter)
for i,r in tr.iterrows():
    w=r.text.split(" ")
    for x in T[r.id]["arguments"]:
        if x["role"] in ("STREET-ARG","PLACE-ARG"): txt2[" ".join(w[x["start"]:x["end"]])][x["role"]]+=1
m2=[(k,v) for k,v in txt2.items() if sum(v.values())>=5]
i2=[(k,v) for k,v in m2 if min(v.values())/sum(v.values())>0.25]
print(f"strings seen >=5 times: {len(m2)}; inconsistent: {len(i2)}", [(k,dict(v)) for k,v in i2[:8]])
