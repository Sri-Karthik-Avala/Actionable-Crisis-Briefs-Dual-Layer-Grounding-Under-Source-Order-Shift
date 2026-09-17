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
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_chunk", type=int, default=4)
ap.add_argument("--ph", default="map")
ap.add_argument("--crf", type=int, default=0)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--warm", type=float, default=0.1)
ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--maxlen", type=int, default=192)
ap.add_argument("--subword", default="first")
ap.add_argument("--lld", type=float, default=1.0)
ap.add_argument("--tag", default="")
ap.add_argument("--full", type=int, default=0)
ap.add_argument("--amp", default="fp16")
ap.add_argument("--cpu", type=int, default=0)
ap.add_argument("--snap", type=int, default=4)
ap.add_argument("--exclude_chunk", type=int, default=-1)
ap.add_argument("--cbal", type=float, default=0.0)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
if args.cpu:
    torch.set_num_threads(4)

tr = pd.read_csv(D + "train.csv")
T = load_targets(D + "train_targets.jsonl")
chunk = np.zeros(len(tr), dtype=int)
for a, b in BLOCKS:
    n = b - a
    for i in range(n):
        chunk[a + i] = min(4, int(i * 5 // n))
val_mask = chunk == args.val_chunk
if args.full:
    val_mask[:] = False
    val_mask[np.where(chunk == args.val_chunk)[0][:50]] = True

ARG_TAGS = ["O"] + [p + r for r in ARGUMENT_ROLES for p in ("B-", "I-")]
ENT_TAGS = ["O"] + [p + t for t in ENTITY_TYPES for p in ("B-", "I-")]
A2I = {t: i for i, t in enumerate(ARG_TAGS)}
E2I = {t: i for i, t in enumerate(ENT_TAGS)}

PH = {"<USER>": "@user", "<URL>": "http", "<EMAIL>": "email", "<PHONE>": "nomor", "<EMPTY>": "_"}


def words_of(text):
    w = text.split(" ")
    if args.ph == "map":
        w = [PH.get(x, x) for x in w]
    elif args.ph == "raw":
        w = [x if x else "_" for x in w]
    return w


def bio(n, spans, key, M):
    y = [0] * n
    for s in spans:
        lab = s[key]
        y[s["start"]] = M["B-" + lab]
        for k in range(s["start"] + 1, s["end"]):
            y[k] = M["I-" + lab]
    return y


tok = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True) if "roberta-base" == args.model else AutoTokenizer.from_pretrained(args.model)


def encode(words):
    ids = [tok.cls_token_id]
    first = []
    last = []
    for w in words:
        p = tok(w, add_special_tokens=False)["input_ids"]
        if not p:
            p = [tok.unk_token_id]
        first.append(len(ids))
        ids.extend(p)
        last.append(len(ids) - 1)
    ids = ids[: args.maxlen - 1] + [tok.sep_token_id]
    first = [min(f, args.maxlen - 2) for f in first]
    last = [min(f, args.maxlen - 2) for f in last]
    return ids, first, last


rows = []
for i, r in tr.iterrows():
    w = words_of(r.text)
    t = T[r.id]
    ids, first, last = encode(w)
    rows.append(dict(ids=ids, first=first, last=last, n=len(w), ya=bio(len(w), t["arguments"], "role", A2I), ye=bio(len(w), t["entities"], "type", E2I), tgt=t))
tr_idx = np.where(~val_mask & (chunk != args.exclude_chunk))[0]
va_idx = np.where(val_mask)[0]
print("train", len(tr_idx), "val", len(va_idx), "max subwords", max(len(r["ids"]) for r in rows), flush=True)


def allowed(tags):
    K = len(tags)
    M = np.zeros((K, K), dtype=bool)
    st = np.zeros(K, dtype=bool)
    for i, a in enumerate(tags):
        for j, b in enumerate(tags):
            if b.startswith("I-"):
                M[i, j] = a != "O" and a[2:] == b[2:]
            else:
                M[i, j] = True
        st[i] = not a.startswith("I-")
    return M, st


class CRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        K = len(tags)
        M, st = allowed(tags)
        self.register_buffer("mask_t", torch.tensor(np.where(M, 0.0, -1e4), dtype=torch.float32))
        self.register_buffer("mask_s", torch.tensor(np.where(st, 0.0, -1e4), dtype=torch.float32))
        self.trans = nn.Parameter(torch.zeros(K, K))
        self.start = nn.Parameter(torch.zeros(K))

    def nll(self, em, y, m):
        B, L, K = em.shape
        tr_ = self.trans + self.mask_t
        s0 = self.start + self.mask_s
        score = s0[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        alpha = s0[None] + em[:, 0]
        for t in range(1, L):
            mt = m[:, t]
            e = em[:, t].gather(1, y[:, t:t + 1]).squeeze(1) + tr_[y[:, t - 1], y[:, t]]
            score = score + e * mt
            na = torch.logsumexp(alpha[:, :, None] + tr_[None] + em[:, t, None, :], dim=1)
            alpha = torch.where(mt[:, None] > 0, na, alpha)
        return (torch.logsumexp(alpha, 1) - score).mean()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(args.model).float()
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(args.drop)
        self.ha = nn.Linear(H, len(ARG_TAGS))
        self.he = nn.Linear(H, len(ENT_TAGS))
        if args.crf:
            self.ca = CRF(ARG_TAGS)
            self.ce = CRF(ENT_TAGS)

    def forward(self, ids, att, first, last, wm):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        if args.subword == "first":
            g = h.gather(1, first[:, :, None].expand(-1, -1, h.size(-1)))
        else:
            g = 0.5 * (h.gather(1, first[:, :, None].expand(-1, -1, h.size(-1))) + h.gather(1, last[:, :, None].expand(-1, -1, h.size(-1))))
        g = self.drop(g)
        return self.ha(g).float(), self.he(g).float()


def batchify(idx):
    B = [rows[i] for i in idx]
    L = max(len(b["ids"]) for b in B)
    W = max(b["n"] for b in B)
    ids = torch.full((len(B), L), tok.pad_token_id)
    att = torch.zeros((len(B), L), dtype=torch.long)
    first = torch.zeros((len(B), W), dtype=torch.long)
    last = torch.zeros((len(B), W), dtype=torch.long)
    wm = torch.zeros((len(B), W))
    ya = torch.zeros((len(B), W), dtype=torch.long)
    ye = torch.zeros((len(B), W), dtype=torch.long)
    for k, b in enumerate(B):
        ids[k, : len(b["ids"])] = torch.tensor(b["ids"])
        att[k, : len(b["ids"])] = 1
        first[k, : b["n"]] = torch.tensor(b["first"])
        last[k, : b["n"]] = torch.tensor(b["last"])
        wm[k, : b["n"]] = 1
        ya[k, : b["n"]] = torch.tensor(b["ya"])
        ye[k, : b["n"]] = torch.tensor(b["ye"])
    return [x.to(dev) for x in (ids, att, first, last, wm, ya, ye)]


def viterbi(lp, tags, trans=None, start=None):
    M, st = allowed(tags)
    tm = np.where(M, 0.0, -1e9) + (0 if trans is None else trans)
    sm = np.where(st, 0.0, -1e9) + (0 if start is None else start)
    n, K = lp.shape
    dp = sm + lp[0]
    bp = np.zeros((n, K), dtype=int)
    for t in range(1, n):
        c = dp[:, None] + tm
        bp[t] = c.argmax(0)
        dp = c.max(0) + lp[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(int(bp[t, path[-1]]))
    return path[::-1]


def spans_of(path, tags, key):
    out = []
    cur = None
    for i, p in enumerate(path + [0]):
        t = tags[p] if i < len(path) else "O"
        if cur is not None and not (t.startswith("I-") and t[2:] == cur[0]):
            out.append({key: cur[0], "start": cur[1], "end": i})
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


WTA = None
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
model = Net().to(dev)
AMP_ON = dev.type == "cuda" and args.amp != "none"
AMP_DT = torch.bfloat16 if args.amp == "bf16" else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=AMP_ON and args.amp == "fp16")
enc_params = [(n, p) for n, p in model.named_parameters() if n.startswith("enc.")]
nl = model.enc.config.num_hidden_layers
groups = []
for n, p in enc_params:
    depth = nl
    if "embeddings" in n:
        depth = 0
    else:
        for k in range(nl):
            if f"layer.{k}." in n:
                depth = k + 1
    scale = args.lld ** (nl - depth)
    groups.append({"params": [p], "lr": args.lr * scale, "weight_decay": 0.0 if ("bias" in n or "LayerNorm" in n) else args.wd})
head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
groups.append({"params": head, "lr": args.head_lr, "weight_decay": 0.0})
opt = torch.optim.AdamW(groups)
base_lrs = [g["lr"] for g in groups]
steps_per = math.ceil(len(tr_idx) / args.bs)
total = steps_per * args.epochs
warm = int(args.warm * total)


def set_lr(step):
    f = step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * f


def predict(idx):
    model.eval()
    outa, oute = [], []
    order = sorted(idx, key=lambda i: len(rows[i]["ids"]))
    res = {}
    with torch.no_grad():
        for s in range(0, len(order), 64):
            bi = order[s: s + 64]
            ids, att, first, last, wm, ya, ye = batchify(bi)
            with torch.autocast(device_type=dev.type, dtype=AMP_DT, enabled=AMP_ON):
                la, le = model(ids, att, first, last, wm)
            la = F.log_softmax(la, -1).cpu().numpy()
            le = F.log_softmax(le, -1).cpu().numpy()
            for k, i in enumerate(bi):
                n = rows[i]["n"]
                res[i] = (la[k, :n], le[k, :n])
    return res


def decode(res, idx):
    ta = tb = sa = sb = None
    if args.crf:
        ta = model.ca.trans.detach().cpu().numpy(); sa = model.ca.start.detach().cpu().numpy()
        tb = model.ce.trans.detach().cpu().numpy(); sb = model.ce.start.detach().cpu().numpy()
    preds = []
    for i in idx:
        la, le = res[i]
        pa = viterbi(la, ARG_TAGS, ta, sa)
        pe = viterbi(le, ENT_TAGS, tb, sb)
        preds.append({"n_tokens": rows[i]["n"], "arguments": spans_of(pa, ARG_TAGS, "role"), "entities": spans_of(pe, ENT_TAGS, "type")})
    return preds


name = args.tag or f"{args.model.split('/')[-1]}_s{args.seed}_c{args.val_chunk}"
os.makedirs(D + "dev/out", exist_ok=True)
step = 0
t0 = time.time()
hist = []
best = -1
rng = np.random.RandomState(args.seed)
for ep in range(args.epochs):
    model.train()
    perm = rng.permutation(tr_idx)
    mb = [perm[s: s + args.bs * 8] for s in range(0, len(perm), args.bs * 8)]
    batches = []
    for m in mb:
        m = sorted(m, key=lambda i: len(rows[i]["ids"]))
        batches += [m[s: s + args.bs] for s in range(0, len(m), args.bs)]
    rng.shuffle(batches)
    tl = 0.0
    for bi in batches:
        ids, att, first, last, wm, ya, ye = batchify(bi)
        set_lr(step)
        with torch.autocast(device_type=dev.type, dtype=AMP_DT, enabled=AMP_ON):
            la, le = model(ids, att, first, last, wm)
        if args.crf:
            loss = model.ca.nll(la, ya, wm) + model.ce.nll(le, ye, wm)
        else:
            m = wm.reshape(-1) > 0
            loss = F.cross_entropy(la.reshape(-1, la.size(-1))[m], ya.reshape(-1)[m], weight=WTA) + F.cross_entropy(le.reshape(-1, le.size(-1))[m], ye.reshape(-1)[m], weight=WTE)
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
