import pandas as pd, numpy as np
D="C:/Users/srika/Downloads/eris_actionable/"
for name in ["train","test"]:
    df=pd.read_csv(D+name+".csv")
    kws=["banjir","kebakaran","gempa","kecelakaan"]
    lab=[]
    for t in df.text:
        hits=[k for k in kws if k in t]
        lab.append(hits[0][0] if len(hits)==1 else ("m" if hits else "-"))
    s="".join(lab)
    print(name, len(s))
    for i in range(0,len(s),200): print("  ",s[i:i+200])
    import collections
    print(collections.Counter(lab))
