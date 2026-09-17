import sys
sys.path.insert(0,"dev")
from ens import *
role=sys.argv[1]; paths=sys.argv[2:]
res=load(paths); keys=sorted(res.keys()); refs=refs_of(keys); preds=decode(res,keys)
k=0
for key,p,r in zip(keys,preds,refs):
    G=[(a["start"],a["end"]) for a in r["arguments"] if a["role"]==role]
    P=[(a["start"],a["end"]) for a in p["arguments"] if a["role"]==role]
    if not G and not P: continue
    if set(G)==set(P): continue
    w=tr.text[key].split(" ")
    print("\n#",key, " ".join(w)[:260])
    print("  GOLD:",[" ".join(w[s:e]) for s,e in G], "| other gold at pred:",[(a["role"]," ".join(w[a["start"]:a["end"]])) for a in r["arguments"] if a["role"]!=role and any(max(0,min(e,a["end"])-max(s,a["start"]))>0 for s,e in P)])
    print("  PRED:",[" ".join(w[s:e]) for s,e in P])
    k+=1
    if k>=int(30): break
