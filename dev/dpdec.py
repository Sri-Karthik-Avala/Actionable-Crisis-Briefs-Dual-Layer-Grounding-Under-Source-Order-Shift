import sys, pickle
sys.path.insert(0,"dev")
from mix import *

def dp_select(P, key, labels, thr):
    n=P.shape[0]; L=P.shape[1]
    lab=P[:,:,1:].argmax(-1); pr=np.take_along_axis(P[:,:,1:],lab[:,:,None],-1)[:,:,0]
    NEG=-1e18
    dp=np.zeros(n+1); choice=[None]*(n+1)
    for i in range(n-1,-1,-1):
        best=dp[i+1]; ch=None
        for l in range(min(L,n-i)):
            v=pr[i,l]-thr
            if v>0:
                tot=v+dp[i+l+1]
                if tot>best: best=tot; ch=l
        dp[i]=best; choice[i]=ch
    out=[]; i=0
    while i<n:
        if choice[i] is None: i+=1
        else:
            l=choice[i]; out.append({key:labels[lab[i,l]],"start":int(i),"end":int(i+l+1)}); i+=l+1
    return out

keys,PA,PE=pickle.load(open("dev/out_mix6.pkl","rb")); refs=refs_of(keys)
for thr in [0.3,0.3544,0.4,0.45]:
    g=[{"arguments":greedy_prob(PA[k],"role",ARGUMENT_ROLES,thr),"entities":greedy_prob(PE[k],"type",ENTITY_TYPES,thr)} for k in keys]
    d=[{"arguments":dp_select(PA[k],"role",ARGUMENT_ROLES,thr),"entities":dp_select(PE[k],"type",ENTITY_TYPES,thr)} for k in keys]
    print("thr",thr,"greedy %.4f"%score_docs(g,refs)[0],"dp %.4f"%score_docs(d,refs)[0],flush=True)
