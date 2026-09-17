p="dev/tagger.py"
s=open(p,encoding="utf-8").read()
s=s.replace('ap.add_argument("--exclude_chunk", type=int, default=-1)','ap.add_argument("--exclude_chunk", type=int, default=-1)\nap.add_argument("--cbal", type=float, default=0.0)',1)
anchor="model = Net().to(dev)"
ins='''WTA = None
WTE = None
if args.cbal:
    ca = np.ones(len(ARGUMENT_ROLES)); ce_ = np.ones(len(ENTITY_TYPES))
    for i in tr_idx:
        for sp in rows[i]["tgt"]["arguments"]: ca[ARGUMENT_ROLES.index(sp["role"])] += 1
        for sp in rows[i]["tgt"]["entities"]: ce_[ENTITY_TYPES.index(sp["type"])] += 1
    wa = np.clip((np.median(ca) / ca) ** args.cbal, 0.3, 4.0)
    we = np.clip((np.median(ce_) / ce_) ** args.cbal, 0.3, 4.0)
    WTA = torch.tensor([1.0] + [float(wa[j // 2]) for j in range(2 * len(ARGUMENT_ROLES))], dtype=torch.float32, device=dev)
    WTE = torch.tensor([1.0] + [float(we[j // 2]) for j in range(2 * len(ENTITY_TYPES))], dtype=torch.float32, device=dev)
    print("bio class weights", dict(zip(ARGUMENT_ROLES, np.round(wa, 2))), flush=True)
model = Net().to(dev)'''
assert s.count(anchor)==1
s=s.replace(anchor,ins,1)
old="            loss = F.cross_entropy(la.reshape(-1, la.size(-1))[m], ya.reshape(-1)[m]) + F.cross_entropy(le.reshape(-1, le.size(-1))[m], ye.reshape(-1)[m])"
new="            loss = F.cross_entropy(la.reshape(-1, la.size(-1))[m], ya.reshape(-1)[m], weight=WTA) + F.cross_entropy(le.reshape(-1, le.size(-1))[m], ye.reshape(-1)[m], weight=WTE)"
assert s.count(old)==1
s=s.replace(old,new)
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("patched tagger cbal")
