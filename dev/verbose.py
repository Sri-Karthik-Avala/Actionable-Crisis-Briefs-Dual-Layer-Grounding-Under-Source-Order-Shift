import sys
sys.path.insert(0,"dev")
from ens import *
res=load(sys.argv[1:]); keys=sorted(res.keys()); refs=refs_of(keys)
score_docs(decode(res,keys),refs,verbose=True)
