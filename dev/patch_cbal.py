p="dev/spanner.py"
s=open(p,encoding="utf-8").read()
s=s.replace('ap.add_argument("--wloss", type=float, default=0.0)','ap.add_argument("--wloss", type=float, default=0.0)\nap.add_argument("--cbal", type=float, default=0.0)',1)
old="""WA = torch.tensor([1.0] + [AW[r] ** args.wloss for r in ARGUMENT_ROLES], dtype=torch.float32, device=dev) if args.wloss else None
WE = torch.tensor([1.0] + [EW[t] ** args.wloss for t in ENTITY_TYPES], dtype=torch.float32, device=dev) if args.wloss else None"""
new="""WA = torch.tensor([1.0] + [AW[r] ** args.wloss for r in ARGUMENT_ROLES], dtype=torch.float32, device=dev) if args.wloss else None
WE = torch.tensor([1.0] + [EW[t] ** args.wloss for t in ENTITY_TYPES], dtype=torch.float32, device=dev) if args.wloss else None
if args.cbal:
    fa = np.ones(len(ARGUMENT_ROLES)); fe = np.ones(len(ENTITY_TYPES))
    for i in tr_idx:
        for s_, l_, c_ in rows[i]["ga"]: fa[c_ - 1] += 1
        for s_, l_, c_ in rows[i]["ge"]: fe[c_ - 1] += 1
    wa = (np.median(fa) / fa) ** args.cbal
    we = (np.median(fe) / fe) ** args.cbal
    WA = torch.tensor([1.0] + list(np.clip(wa, 0.3, 4.0)), dtype=torch.float32, device=dev)
    WE = torch.tensor([1.0] + list(np.clip(we, 0.3, 4.0)), dtype=torch.float32, device=dev)
    print("class weights arg", dict(zip(ARGUMENT_ROLES, np.round(WA.cpu().numpy()[1:], 2))), flush=True)"""
assert s.count(old)==1
s=s.replace(old,new)
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("patched cbal")
