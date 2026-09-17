import argparse, json, math, os, random, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abgs_metric import ARGUMENT_ROLES, ENTITY_TYPES, score_docs, load_targets

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
BLOCKS = [(0, 1000), (1000, 1724), (1724, 2626), (2626, 3318)]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="xlm-roberta-base")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_chunk", type=int, default=4)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--warm", type=float, default=0.1)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--maxlen", type=int, default=192)
ap.add_argument("--maxspan", type=int, default=20)
ap.add_argument("--hd", type=int, default=256)
ap.add_argument("--tag", default="")
ap.add_argument("--amp", default="fp16")
ap.add_argument("--full", type=int, default=0)
ap.add_argument("--cpu", type=int, default=0)
ap.add_argument("--snap", type=int, default=4)
ap.add_argument("--wloss", type=float, default=0.0)
ap.add_argument("--cbal", type=float, default=0.0)
ap.add_argument("--exclude_chunk", type=int, default=-1)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
if args.cpu:
    torch.set_num_threads(4)

tr = pd.read_csv(D + "train.csv")
T = load_targets(D + "train_targets.jsonl")
chunk = np.zeros(len(tr), dtype=int)
for a, b in BLOCKS:
    for i in range(b - a):
        chunk[a + i] = min(4, int(i * 5 // (b - a)))
val_mask = chunk == args.val_chunk
PH = {"<USER>": "@user", "<URL>": "http", "<EMAIL>": "email", "<PHONE>": "nomor", "<EMPTY>": "_"}
A2I = {r: i + 1 for i, r in enumerate(ARGUMENT_ROLES)}
E2I = {t: i + 1 for i, t in enumerate(ENTITY_TYPES)}
LS = args.maxspan

tok = AutoTokenizer.from_pretrained(args.model)


def encode(words):
    ids = [tok.cls_token_id]
    first = []
    for w in words:
        p = tok(w, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
        first.append(len(ids))
        ids.extend(p)
    ids = ids[: args.maxlen - 1] + [tok.sep_token_id]
    first = [min(f, args.maxlen - 2) for f in first]
    return ids, first


rows = []
dropped = 0
for i, r in tr.iterrows():
    w = [PH.get(x, x) for x in r.text.split(" ")]
    t = T[r.id]
    ids, first = encode(w)
    ga = [(s["start"], s["end"] - s["start"] - 1, A2I[s["role"]]) for s in t["arguments"] if s["end"] - s["start"] <= LS]
    ge = [(s["start"], s["end"] - s["start"] - 1, E2I[s["type"]]) for s in t["entities"] if s["end"] - s["start"] <= LS]
    dropped += len(t["arguments"]) + len(t["entities"]) - len(ga) - len(ge)
    rows.append(dict(ids=ids, first=first, n=len(w), ga=ga, ge=ge, tgt=t))
tr_idx = np.where(~val_mask & (chunk != args.exclude_chunk))[0]
va_idx = np.where(val_mask)[0]
print("train", len(tr_idx), "val", len(va_idx), "dropped long spans", dropped, flush=True)


class SpanHead(nn.Module):
    def __init__(self, H, K):
        super().__init__()
        d = args.hd
        self.s = nn.Sequential(nn.Linear(H, d), nn.GELU(), nn.Dropout(args.drop))
        self.e = nn.Sequential(nn.Linear(H, d), nn.GELU(), nn.Dropout(args.drop))
        self.U = nn.Parameter(torch.randn(K + 1, d, d) * 0.01)
        self.ws = nn.Linear(d, K + 1)
        self.we = nn.Linear(d, K + 1, bias=False)
        self.len_emb = nn.Embedding(LS, K + 1)

    def forward(self, g):
        s = self.s(g).float()
        e = self.e(g).float()
        B, W, _ = s.shape
        L = min(LS, W)
        idx_end = torch.arange(W, device=g.device)[:, None] + torch.arange(L, device=g.device)[None]
        valid = idx_end < W
        idx_c = idx_end.clamp(max=W - 1)
        ee = e[:, idx_c]
        bil = torch.einsum("bid,kde,bile->bilk", s, self.U, ee)
        out = bil + self.ws(s)[:, :, None, :] + self.we(ee) + self.len_emb.weight[None, None, :L, :]
        return out, valid


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model).float()
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(args.drop)
        self.ha = SpanHead(H, len(ARGUMENT_ROLES))
        self.he = SpanHead(H, len(ENTITY_TYPES))

    def forward(self, ids, att, first):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        g = self.drop(h.gather(1, first[:, :, None].expand(-1, -1, h.size(-1))))
        return self.ha(g), self.he(g)


def batchify(idx):
    B = [rows[i] for i in idx]
    L = max(len(b["ids"]) for b in B)
    W = max(b["n"] for b in B)
    Ls = min(LS, W)
    ids = torch.full((len(B), L), tok.pad_token_id)
    att = torch.zeros((len(B), L), dtype=torch.long)
    first = torch.zeros((len(B), W), dtype=torch.long)
    n = torch.zeros(len(B), dtype=torch.long)
    ya = torch.zeros((len(B), W, Ls), dtype=torch.long)
    ye = torch.zeros((len(B), W, Ls), dtype=torch.long)
    for k, b in enumerate(B):
        ids[k, : len(b["ids"])] = torch.tensor(b["ids"])
        att[k, : len(b["ids"])] = 1
        first[k, : b["n"]] = torch.tensor(b["first"])
        n[k] = b["n"]
        for s, l, c in b["ga"]:
            ya[k, s, l] = c
        for s, l, c in b["ge"]:
            ye[k, s, l] = c
    return [x.to(dev) for x in (ids, att, first, n, ya, ye)]


def span_mask(n, W, L):
    st = torch.arange(W, device=dev)[None, :, None]
    ln = torch.arange(L, device=dev)[None, None, :]
    return (st + ln) < n[:, None, None]


def greedy(lp, key, labels):
    n, L, K1 = lp.shape
    best_lab = lp[:, :, 1:].argmax(-1) + 1
    best_lp = np.take_along_axis(lp, best_lab[:, :, None], -1)[:, :, 0]
    cand = []
    for s in range(n):
        for l in range(min(L, n - s)):
            if best_lp[s, l] > lp[s, l, 0]:
                cand.append((best_lp[s, l], s, l, best_lab[s, l]))
    cand.sort(key=lambda x: -x[0])
    used = np.zeros(n, dtype=bool)
    out = []
    for sc, s, l, c in cand:
        if not used[s: s + l + 1].any():
            used[s: s + l + 1] = True
            out.append({key: labels[c - 1], "start": int(s), "end": int(s + l + 1)})
    out.sort(key=lambda x: x["start"])
    return out


AW = {"PLACE-ARG":2.0,"STREET-ARG":2.0,"TIME-ARG":1.6,"AFFECTEDOBJECTS-ARG":1.6,"DEATHVICTIM-ARG":2.4,"WOUNDVICTIM-ARG":2.4,"OFFICER-ARG":1.0,"REASON-ARG":1.2,"INFORMATION-ARG":1.0,"FIRE-EVENT":1.2,"FLOOD-EVENT":1.2,"EARTHQUAKE-EVENT":1.2,"ACCIDENT-EVENT":1.2,"FALSE-EVENT":0.8}
EW = {"LOC":1.5,"PLOC":1.5,"EVE":1.0,"ORG":1.0,"ARG":0.8}
WA = torch.tensor([1.0] + [AW[r] ** args.wloss for r in ARGUMENT_ROLES], dtype=torch.float32, device=dev) if args.wloss else None
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
    print("class weights arg", dict(zip(ARGUMENT_ROLES, np.round(WA.cpu().numpy()[1:], 2))), flush=True)
model = Net().to(dev)
AMP_ON = dev.type == "cuda" and args.amp != "none"
AMP_DT = torch.bfloat16 if args.amp == "bf16" else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=AMP_ON and args.amp == "fp16")
enc = [p for n_, p in model.named_parameters() if n_.startswith("enc.")]
head = [p for n_, p in model.named_parameters() if not n_.startswith("enc.")]
opt = torch.optim.AdamW([{"params": enc, "lr": args.lr, "weight_decay": args.wd}, {"params": head, "lr": args.head_lr, "weight_decay": 0.0}])
base_lrs = [g["lr"] for g in opt.param_groups]
total = math.ceil(len(tr_idx) / args.bs) * args.epochs
warm = int(args.warm * total)


def predict(idx):
    model.eval()
    res = {}
    order = sorted(idx, key=lambda i: len(rows[i]["ids"]))
    with torch.no_grad():
        for s in range(0, len(order), 64):
            bi = order[s: s + 64]
            ids, att, first, n, _, _ = batchify(bi)
            with torch.autocast(device_type=dev.type, dtype=AMP_DT, enabled=AMP_ON):
                (oa, va), (oe, ve) = model(ids, att, first)
            la = F.log_softmax(oa.float(), -1).cpu().numpy()
            le = F.log_softmax(oe.float(), -1).cpu().numpy()
            for k, i in enumerate(bi):
                m = rows[i]["n"]
                res[i] = (la[k, :m], le[k, :m])
    return res


def decode(res, idx):
    return [{"arguments": greedy(res[i][0], "role", ARGUMENT_ROLES), "entities": greedy(res[i][1], "type", ENTITY_TYPES)} for i in idx]


name = args.tag or "span"
os.makedirs(D + "dev/out", exist_ok=True)
step = 0
t0 = time.time()
hist = []
best = -1
rng = np.random.RandomState(args.seed)
for ep in range(args.epochs):
    model.train()
    perm = rng.permutation(tr_idx)
    batches = []
    for s in range(0, len(perm), args.bs * 8):
        m = sorted(perm[s: s + args.bs * 8], key=lambda i: len(rows[i]["ids"]))
        batches += [m[c: c + args.bs] for c in range(0, len(m), args.bs)]
    rng.shuffle(batches)
    tl = 0.0
    for bi in batches:
        ids, att, first, n, ya, ye = batchify(bi)
        f = step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        with torch.autocast(device_type=dev.type, dtype=AMP_DT, enabled=AMP_ON):
            (oa, va), (oe, ve) = model(ids, att, first)
        W = oa.size(1); L = oa.size(2)
        sm = span_mask(n, W, L)
        loss = F.cross_entropy(oa.float()[sm], ya[sm], weight=WA) + F.cross_entropy(oe.float()[sm], ye[sm], weight=WE)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tl += loss.item()
    res = predict(va_idx)
    preds = decode(res, va_idx)
    refs = [rows[i]["tgt"] for i in va_idx]
    tot, a, e = score_docs(preds, refs)
    hist.append((ep, tl / len(batches), tot, a, e))
    print(f"ep {ep} loss {tl / len(batches):.4f} val {tot:.4f} arg {a:.4f} ent {e:.4f} t {time.time() - t0:.0f}s", flush=True)
    if ep >= args.epochs - args.snap or ep == args.epochs - 1:
        dump = {int(i): (res[i][0].astype(np.float16), res[i][1].astype(np.float16)) for i in va_idx}
        np.save(D + f"dev/out/{name}_ep{ep}.npy", dump, allow_pickle=True)
        if ep == args.epochs - 1:
            np.save(D + f"dev/out/{name}_last.npy", dump, allow_pickle=True)
score_docs(preds, refs, verbose=True)
with open(D + "dev/out/results.jsonl", "a") as f:
    f.write(json.dumps({"name": name, "args": vars(args), "hist": hist, "sec": time.time() - t0}) + "\n")
