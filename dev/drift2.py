import sys, collections, pickle
import numpy as np, pandas as pd
sys.path.insert(0,"dev")
from mix import *
D="C:/Users/srika/Downloads/eris_actionable/"
tr=pd.read_csv(D+"train.csv"); te=pd.read_csv(D+"test.csv"); T=load_targets(D+"train_targets.jsonl")
BLOCKS=[(0,1000),(1000,1724),(1724,2626),(2626,3318)]
chunk=np.zeros(len(tr),int)
for a,z in BLOCKS:
    for i in range(z-a): chunk[a+i]=min(4,int(i*5//(z-a)))
print("=== per-chunk label rate PER TOKEN (x1000), chunks 0..4")
for lab in ARGUMENT_ROLES+ENTITY_TYPES:
    key="arguments" if lab in ARGUMENT_ROLES else "entities"; fld="role" if key=="arguments" else "type"
    row=[]
    for c in range(5):
        idx=[i for i in range(len(tr)) if chunk[i]==c]
        n=sum(len([x for x in T[tr.id[i]][key] if x[fld]==lab]) for i in idx)
        toks=sum(tr.n_tokens[i] for i in idx)
        row.append(1000*n/toks)
    trend="UP" if row[4]>row[0] else "down"
    print(f"  {lab:22s} "+" ".join(f"{v:6.2f}" for v in row)+f"   {trend}")
print("\n=== per-doc vs per-token prior targets for the holdout (chunk 4)")
fit=[i for i in range(len(tr)) if chunk[i]<4]; hold=[i for i in range(len(tr)) if chunk[i]==4]
tok_fit=sum(tr.n_tokens[i] for i in fit); tok_hold=sum(tr.n_tokens[i] for i in hold)
for lab in ARGUMENT_ROLES[:6]:
    nf=sum(len([x for x in T[tr.id[i]]["arguments"] if x["role"]==lab]) for i in fit)
    nh=sum(len([x for x in T[tr.id[i]]["arguments"] if x["role"]==lab]) for i in hold)
    per_doc=nf/len(fit)*len(hold); per_tok=nf/tok_fit*tok_hold
    print(f"  {lab:22s} actual {nh:4d}  per-doc pred {per_doc:6.1f}  per-token pred {per_tok:6.1f}")
print(f"\ntokens: fit {tok_fit} hold {tok_hold} ({tok_hold/len(hold):.1f}/doc) test {te.n_tokens.sum()} ({te.n_tokens.mean():.1f}/doc)")
print("per-doc scaling would predict test spans =", round(sum(len(T[tr.id[i]]['arguments']) for i in range(len(tr)))/len(tr)*len(te)))
print("per-token scaling would predict test spans =", round(sum(len(T[tr.id[i]]['arguments']) for i in range(len(tr)))/sum(tr.n_tokens)*te.n_tokens.sum()))
