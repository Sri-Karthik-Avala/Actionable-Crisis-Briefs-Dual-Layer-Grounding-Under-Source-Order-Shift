import re
for p,istag in [("dev/tagger.py",True),("dev/spanner.py",False)]:
    s=open(p,encoding="utf-8").read()
    old_t = """    if ep == args.epochs - 1 or tot > best:
        best = max(best, tot)
        dump = {int(i): (res[i][0].astype(np.float16), res[i][1].astype(np.float16)) for i in va_idx}
        tag = "last" if ep == args.epochs - 1 else "best"
        np.save(D + f"dev/out/{name}_{tag}.npy", dump, allow_pickle=True)"""
    old_s = """    if ep == args.epochs - 1 or tot > best:
        best = max(best, tot)
        dump = {int(i): (res[i][0].astype(np.float16), res[i][1].astype(np.float16)) for i in va_idx}
        np.save(D + f"dev/out/{name}_{'last' if ep == args.epochs - 1 else 'best'}.npy", dump, allow_pickle=True)"""
    new = """    if ep >= args.epochs - args.snap or ep == args.epochs - 1:
        dump = {int(i): (res[i][0].astype(np.float16), res[i][1].astype(np.float16)) for i in va_idx}
        np.save(D + f"dev/out/{name}_ep{ep}.npy", dump, allow_pickle=True)
        if ep == args.epochs - 1:
            np.save(D + f"dev/out/{name}_last.npy", dump, allow_pickle=True)"""
    old = old_t if istag else old_s
    assert s.count(old)==1, p
    s=s.replace(old,new)
    s=s.replace('ap.add_argument("--cpu", type=int, default=0)','ap.add_argument("--cpu", type=int, default=0)\nap.add_argument("--snap", type=int, default=4)',1)
    open(p,"w",encoding="utf-8",newline="\n").write(s)
    print("patched",p)
