import pandas as pd, numpy as np, json
D="C:/Users/srika/Downloads/eris_actionable/"
for name in ["train","test"]:
    df=pd.read_csv(D+name+".csv")
    kws={"banjir":"b","gempa":"g","kebakaran":"f","terbakar":"f","kecelakaan":"a"}
    lab=[]
    for t in df.text:
        hits={v for k,v in kws.items() if k in t}
        lab.append(hits.pop() if len(hits)==1 else "-")
    lab=np.array(lab)
    order=[]
    n=len(lab)
    best=None
    cats="bgfa"
    import itertools
    for perm in itertools.permutations(cats):
        # DP over 3 boundaries for this order: maximize agreement
        pref={c:np.concatenate([[0],np.cumsum(lab==c)]) for c in cats}
        # brute force boundaries via cumulative scores
        s0=pref[perm[0]]; s1=pref[perm[1]]; s2=pref[perm[2]]; s3=pref[perm[3]]
        # f(i,j,k)= s0[i] + s1[j]-s1[i] + s2[k]-s2[j] + s3[n]-s3[k]
        A=s0-s1  # choose i
        bestv=-1;arg=None
        # O(n^2) with prefix max
        bi=np.maximum.accumulate(A); bia=np.array([np.argmax(A[:j+1]) for j in range(n+1)])
        B=bi+s1-s2
        bj=np.maximum.accumulate(B); bja=np.array([np.argmax(B[:k+1]) for k in range(n+1)])
        C=bj+s2-s3
        k=int(np.argmax(C)); v=C[k]+s3[n]
        j=int(bja[k]); i=int(bia[j])
        if best is None or v>best[0]: best=(v,perm,(i,j,k))
    v,perm,(i,j,k)=best
    print(name,"order",perm,"bounds",(0,i,j,k,n),"agree",v,"of",int((lab!='-').sum()))
    print("  sizes",[i,j-i,k-j,n-k])
