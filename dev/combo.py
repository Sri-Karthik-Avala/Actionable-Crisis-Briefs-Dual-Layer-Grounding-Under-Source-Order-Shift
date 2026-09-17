import sys, itertools
sys.path.insert(0,"dev")
from ens import *
O="colab_out/out/"
names=sys.argv[1].split(",")
suffix=sys.argv[2] if len(sys.argv)>2 else "last"
for k in range(1,len(names)+1):
    for c in itertools.combinations(names,k):
        res=load([O+f"{n}_{suffix}.npy" for n in c]); keys=sorted(res.keys())
        print(k, "+".join(c), "%.4f %.4f %.4f"%score_docs(decode(res,keys),refs_of(keys)), flush=True)
