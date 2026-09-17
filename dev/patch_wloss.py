wts = {"PLACE-ARG":2.0,"STREET-ARG":2.0,"TIME-ARG":1.6,"AFFECTEDOBJECTS-ARG":1.6,"DEATHVICTIM-ARG":2.4,"WOUNDVICTIM-ARG":2.4,"OFFICER-ARG":1.0,"REASON-ARG":1.2,"INFORMATION-ARG":1.0,"FIRE-EVENT":1.2,"FLOOD-EVENT":1.2,"EARTHQUAKE-EVENT":1.2,"ACCIDENT-EVENT":1.2,"FALSE-EVENT":0.8}
ent = {"LOC":1.5,"PLOC":1.5,"EVE":1.0,"ORG":1.0,"ARG":0.8}
p="dev/spanner.py"
s=open(p,encoding="utf-8").read()
s=s.replace('ap.add_argument("--snap", type=int, default=4)','ap.add_argument("--snap", type=int, default=4)\nap.add_argument("--wloss", type=float, default=0.0)',1)
old="""        loss = F.cross_entropy(oa.float()[sm], ya[sm]) + F.cross_entropy(oe.float()[sm], ye[sm])"""
new="""        loss = F.cross_entropy(oa.float()[sm], ya[sm], weight=WA) + F.cross_entropy(oe.float()[sm], ye[sm], weight=WE)"""
assert s.count(old)==1
s=s.replace(old,new)
anchor="model = Net().to(dev)"
ins = """AW = {"PLACE-ARG":2.0,"STREET-ARG":2.0,"TIME-ARG":1.6,"AFFECTEDOBJECTS-ARG":1.6,"DEATHVICTIM-ARG":2.4,"WOUNDVICTIM-ARG":2.4,"OFFICER-ARG":1.0,"REASON-ARG":1.2,"INFORMATION-ARG":1.0,"FIRE-EVENT":1.2,"FLOOD-EVENT":1.2,"EARTHQUAKE-EVENT":1.2,"ACCIDENT-EVENT":1.2,"FALSE-EVENT":0.8}
EW = {"LOC":1.5,"PLOC":1.5,"EVE":1.0,"ORG":1.0,"ARG":0.8}
WA = torch.tensor([1.0] + [AW[r] ** args.wloss for r in ARGUMENT_ROLES], dtype=torch.float32, device=dev) if args.wloss else None
WE = torch.tensor([1.0] + [EW[t] ** args.wloss for t in ENTITY_TYPES], dtype=torch.float32, device=dev) if args.wloss else None
model = Net().to(dev)"""
assert s.count(anchor)==1
s=s.replace(anchor,ins,1)
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("patched wloss")
