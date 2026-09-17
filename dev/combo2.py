import sys, itertools
sys.path.insert(0,"dev")
from ens import *
O="colab_out/out/"
names=sys.argv[1].split(","); kmin=int(sys.argv[2]); kmax=int(sys.argv[3])
cache={n:np.load(O+f"{n}_last.npy",allow_pickle=True).item() for n in names}
keys=sorted(cache[names[0]].keys()); refs=refs_of(keys)
out=[]
for k in range(kmin,kmax+1):
    for c in itertools.combinations(names,k):
        res={kk:(np.mean([cache[n][kk][0].astype(np.float32) for n in c],0),np.mean([cache[n][kk][1].astype(np.float32) for n in c],0)) for kk in keys}
        s=score_docs(decode(res,keys),refs)
        out.append((s[0],"+".join(c)))
        print(k,"+".join(c),"%.4f %.4f %.4f"%s,flush=True)
print("TOP"); [print("%.4f %s"%x) for x in sorted(out)[-8:]]
