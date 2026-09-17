import sys, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from abgs_metric import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); te=pd.read_csv(D+"test.csv"); T=load_targets(D+"train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int); blk=np.zeros(len(tr),int)
for b,(a,z) in enumerate(BLOCKS):
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a))); blk[a+i]=b
print("=== length: train vs test")
print("train mean %.1f p90 %.0f max %d"%(tr.n_tokens.mean(),np.percentile(tr.n_tokens,90),tr.n_tokens.max()))
print("test  mean %.1f p90 %.0f max %d"%(te.n_tokens.mean(),np.percentile(te.n_tokens,90),te.n_tokens.max()))
for b,name in enumerate(["flood","quake","fire","accident"]):
    m=blk==b
    print(f"  block {name}: train early(0-3) mean {tr.n_tokens[m&(chunk<4)].mean():.1f}  train tail(4) mean {tr.n_tokens[m&(chunk==4)].mean():.1f}")
tb=[(0,250),(250,432),(432,658),(658,832)]
for (a,z),name in zip(tb,["flood","quake","fire","accident"]):
    print(f"  block {name}: TEST mean {te.n_tokens[a:z].mean():.1f}")
print("\n=== label rate per doc: early chunks(0-3) vs tail chunk(4)")
def rates(idx):
    ra=collections.Counter(); re_=collections.Counter()
    for i in idx:
        t=T[tr.id[i]]
        for x in t["arguments"]: ra[x["role"]]+=1
        for x in t["entities"]: re_[x["type"]]+=1
    n=len(idx)
    return {k:v/n for k,v in ra.items()}, {k:v/n for k,v in re_.items()}
early=[i for i in range(len(tr)) if chunk[i]<4]; tail=[i for i in range(len(tr)) if chunk[i]==4]
ra_e,re_e=rates(early); ra_t,re_t=rates(tail)
for k in ARGUMENT_ROLES:
    e,t=ra_e.get(k,0),ra_t.get(k,0)
    print(f"  {k:22s} early {e:.3f} tail {t:.3f}  ratio {t/max(e,1e-9):.2f}")
for k in ENTITY_TYPES:
    e,t=re_e.get(k,0),re_t.get(k,0)
    print(f"  {k:22s} early {e:.3f} tail {t:.3f}  ratio {t/max(e,1e-9):.2f}")
print("\n=== spans per token by doc length (train)")
for lo,hi in [(0,15),(15,25),(25,35),(35,50),(50,200)]:
    idx=[i for i in range(len(tr)) if lo<=tr.n_tokens[i]<hi]
    if not idx: continue
    na=sum(len(T[tr.id[i]]["arguments"]) for i in idx); ne=sum(len(T[tr.id[i]]["entities"]) for i in idx)
    nt=sum(tr.n_tokens[i] for i in idx)
    print(f"  len[{lo},{hi}) docs {len(idx):5d} args/doc {na/len(idx):.2f} args/token {na/nt:.3f} ents/doc {ne/len(idx):.2f}")
print("\ntest length buckets:", {f"[{lo},{hi})":int(((te.n_tokens>=lo)&(te.n_tokens<hi)).sum()) for lo,hi in [(0,15),(15,25),(25,35),(35,50),(50,200)]})
print("train length buckets:", {f"[{lo},{hi})":int(((tr.n_tokens>=lo)&(tr.n_tokens<hi)).sum()) for lo,hi in [(0,15),(15,25),(25,35),(35,50),(50,200)]})
