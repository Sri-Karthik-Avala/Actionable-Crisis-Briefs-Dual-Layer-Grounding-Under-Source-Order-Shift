# made by - Karthik
import sys
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

ARGUMENT_ROLES = ["PLACE-ARG", "STREET-ARG", "TIME-ARG", "AFFECTEDOBJECTS-ARG", "DEATHVICTIM-ARG", "WOUNDVICTIM-ARG", "OFFICER-ARG", "REASON-ARG", "INFORMATION-ARG", "FIRE-EVENT", "FLOOD-EVENT", "EARTHQUAKE-EVENT", "ACCIDENT-EVENT", "FALSE-EVENT"]
ENTITY_TYPES = ["ARG", "EVE", "LOC", "PLOC", "ORG"]

CALIBRATION_MEMBERS = [
    {"backbone": "indolem/indobertweet-base-uncased", "head": "bio", "lr": 5e-5, "epochs": 10, "seed": 21},
    {"backbone": "indolem/indobertweet-base-uncased", "head": "span", "lr": 5e-5, "epochs": 10, "seed": 22},
]
MEMBERS = [
    {"backbone": "FacebookAI/xlm-roberta-large", "head": "bio", "lr": 2e-5, "epochs": 10, "seed": 11},
    {"backbone": "indolem/indobertweet-base-uncased", "head": "span", "lr": 5e-5, "epochs": 10, "seed": 12},
    {"backbone": "microsoft/mdeberta-v3-base", "head": "bio", "lr": 5e-5, "epochs": 10, "seed": 13},
    {"backbone": "indobenchmark/indobert-large-p1", "head": "bio", "lr": 2e-5, "epochs": 10, "seed": 14},
    {"backbone": "indolem/indobertweet-base-uncased", "head": "bio", "lr": 5e-5, "epochs": 10, "seed": 15},
    {"backbone": "FacebookAI/xlm-roberta-large", "head": "span", "lr": 2e-5, "epochs": 10, "seed": 16},
]
SNAPSHOTS = 2
CLASS_BALANCE = 0.3
CLASS_WEIGHT_FLOOR = 0.5
CLASS_WEIGHT_CEIL = 3.0
THRESHOLD_GRID = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
CALIBRATION_SEGMENTS = 5
CALIBRATION_TAIL = 0.2
BATCH_SIZE = 16
HEAD_LR = 1e-3
WARMUP = 0.1
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MAX_SUBWORDS = 320
MAX_SPAN = 20
SPAN_DIM = 150
PLACEHOLDERS = {"<USER>": "@user", "<URL>": "http", "<EMAIL>": "email", "<PHONE>": "nomor", "<EMPTY>": "_"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
amp_dtype = torch.float16
use_amp = device.type == "cuda"

ARG_TAGS = ["O"] + [p + r for r in ARGUMENT_ROLES for p in ("B-", "I-")]
ENT_TAGS = ["O"] + [p + t for t in ENTITY_TYPES for p in ("B-", "I-")]

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
targets = pd.read_json(public_dir / "train_targets.jsonl", lines=True)
target_of = dict(zip(targets["id"], targets["target"]))
print("train", len(train), "test", len(test), flush=True)


def words_of(text):
    return [PLACEHOLDERS.get(w, w) for w in text.split(" ")]


def bio(n, spans, key, tags):
    index = {t: i for i, t in enumerate(tags)}
    y = [0] * n
    for s in spans:
        y[s["start"]] = index["B-" + s[key]]
        for k in range(s["start"] + 1, s["end"]):
            y[k] = index["I-" + s[key]]
    return y


def span_labels(spans, key, labels):
    index = {t: i + 1 for i, t in enumerate(labels)}
    return [(s["start"], s["end"] - s["start"] - 1, index[s[key]]) for s in spans if s["end"] - s["start"] <= MAX_SPAN]


def encode_rows(tok, frame, with_labels):
    rows = []
    for r in frame.itertuples(index=False):
        words = words_of(r.text)
        ids = [tok.cls_token_id]
        first = []
        for w in words:
            pieces = tok(w, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
            first.append(len(ids))
            ids.extend(pieces)
        ids = ids[: MAX_SUBWORDS - 1] + [tok.sep_token_id]
        first = [min(f, MAX_SUBWORDS - 2) for f in first]
        row = {"ids": ids, "first": first, "n": len(words)}
        if with_labels:
            t = target_of[r.id]
            row["ya"] = bio(len(words), t["arguments"], "role", ARG_TAGS)
            row["ye"] = bio(len(words), t["entities"], "type", ENT_TAGS)
            row["sa"] = span_labels(t["arguments"], "role", ARGUMENT_ROLES)
            row["se"] = span_labels(t["entities"], "type", ENTITY_TYPES)
        rows.append(row)
    return rows


class SpanHead(nn.Module):
    def __init__(self, hidden, n_labels):
        super().__init__()
        self.start = nn.Sequential(nn.Linear(hidden, SPAN_DIM), nn.GELU(), nn.Dropout(DROPOUT))
        self.end = nn.Sequential(nn.Linear(hidden, SPAN_DIM), nn.GELU(), nn.Dropout(DROPOUT))
        self.bilinear = nn.Parameter(torch.randn(n_labels + 1, SPAN_DIM, SPAN_DIM) * 0.01)
        self.start_score = nn.Linear(SPAN_DIM, n_labels + 1)
        self.end_score = nn.Linear(SPAN_DIM, n_labels + 1, bias=False)
        self.length_score = nn.Embedding(MAX_SPAN, n_labels + 1)

    def forward(self, g):
        s = self.start(g).float()
        e = self.end(g).float()
        W = s.size(1)
        L = min(MAX_SPAN, W)
        end_index = (torch.arange(W, device=g.device)[:, None] + torch.arange(L, device=g.device)[None]).clamp(max=W - 1)
        ee = e[:, end_index]
        scores = torch.einsum("bid,kde,bile->bilk", s, self.bilinear, ee)
        return scores + self.start_score(s)[:, :, None, :] + self.end_score(ee) + self.length_score.weight[None, None, :L, :]


class Tagger(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone).float()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.head = head
        if head == "bio":
            self.arg_head = nn.Linear(2 * hidden, len(ARG_TAGS))
            self.ent_head = nn.Linear(2 * hidden, len(ENT_TAGS))
        else:
            self.arg_head = SpanHead(2 * hidden, len(ARGUMENT_ROLES))
            self.ent_head = SpanHead(2 * hidden, len(ENTITY_TYPES))

    def forward(self, ids, att, first):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        words = h.gather(1, first[:, :, None].expand(-1, -1, h.size(-1)))
        context = h[:, :1, :].expand(-1, words.size(1), -1)
        g = self.dropout(torch.cat([words, context], -1))
        return self.arg_head(g).float(), self.ent_head(g).float()


def collate(rows, idx, pad_id, with_labels):
    batch = [rows[i] for i in idx]
    L = max(len(b["ids"]) for b in batch)
    W = max(b["n"] for b in batch)
    S = min(MAX_SPAN, W)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    first = torch.zeros((len(batch), W), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    ya = torch.zeros((len(batch), W), dtype=torch.long)
    ye = torch.zeros((len(batch), W), dtype=torch.long)
    sa = torch.zeros((len(batch), W, S), dtype=torch.long)
    se = torch.zeros((len(batch), W, S), dtype=torch.long)
    for k, b in enumerate(batch):
        ids[k, : len(b["ids"])] = torch.tensor(b["ids"])
        att[k, : len(b["ids"])] = 1
        first[k, : b["n"]] = torch.tensor(b["first"])
        lengths[k] = b["n"]
        if with_labels:
            ya[k, : b["n"]] = torch.tensor(b["ya"])
            ye[k, : b["n"]] = torch.tensor(b["ye"])
            for s, l, c in b["sa"]:
                sa[k, s, l] = c
            for s, l, c in b["se"]:
                se[k, s, l] = c
    return [x.to(device) for x in (ids, att, first, lengths, ya, ye, sa, se)]


def class_weights(rows, head):
    count_a = np.ones(len(ARGUMENT_ROLES))
    count_e = np.ones(len(ENTITY_TYPES))
    for r in rows:
        for _, _, c in r["sa"]:
            count_a[c - 1] += 1
        for _, _, c in r["se"]:
            count_e[c - 1] += 1
    wa = np.clip((np.median(count_a) / count_a) ** CLASS_BALANCE, CLASS_WEIGHT_FLOOR, CLASS_WEIGHT_CEIL)
    we = np.clip((np.median(count_e) / count_e) ** CLASS_BALANCE, CLASS_WEIGHT_FLOOR, CLASS_WEIGHT_CEIL)
    if head == "bio":
        va = [1.0] + [float(wa[j // 2]) for j in range(2 * len(ARGUMENT_ROLES))]
        ve = [1.0] + [float(we[j // 2]) for j in range(2 * len(ENTITY_TYPES))]
    else:
        va = [1.0] + [float(x) for x in wa]
        ve = [1.0] + [float(x) for x in we]
    return torch.tensor(va, dtype=torch.float32, device=device), torch.tensor(ve, dtype=torch.float32, device=device)


def member_loss(model, la, le, lengths, ya, ye, sa, se, weight_a, weight_e):
    W = la.size(1)
    if model.head == "bio":
        wm = torch.arange(W, device=device)[None, :] < lengths[:, None]
        return F.cross_entropy(la[wm], ya[wm], weight=weight_a) + F.cross_entropy(le[wm], ye[wm], weight=weight_e)
    S = la.size(2)
    sm = (torch.arange(W, device=device)[None, :, None] + torch.arange(S, device=device)[None, None, :]) < lengths[:, None, None]
    return F.cross_entropy(la[sm], sa[sm], weight=weight_a) + F.cross_entropy(le[sm], se[sm], weight=weight_e)


def bio_to_span_probs(logp, tags, labels):
    p = np.exp(logp)
    n = p.shape[0]
    K = len(labels)
    pb = p[:, [tags.index("B-" + x) for x in labels]]
    pi = np.concatenate([p[:, [tags.index("I-" + x) for x in labels]], np.zeros((1, K))], 0)
    out = np.zeros((n, MAX_SPAN, K + 1))
    for s in range(n):
        L = min(MAX_SPAN, n - s)
        inner = np.cumprod(np.concatenate([np.ones((1, K)), pi[s + 1: s + L]], 0), 0)
        out[s, :L, 1:] = pb[s][None, :] * inner * (1.0 - pi[s + 1: s + L + 1])
    out[:, :, 0] = np.maximum(0.0, 1.0 - out[:, :, 1:].sum(-1))
    return out


def span_probs(logp):
    out = np.zeros((logp.shape[0], MAX_SPAN, logp.shape[2]))
    out[:, : logp.shape[1]] = np.exp(logp)
    return out


def predict(model, rows, pad_id):
    model.eval()
    out_a = [None] * len(rows)
    out_e = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    with torch.no_grad():
        for s in range(0, len(order), 64):
            idx = order[s: s + 64]
            ids, att, first, lengths, _, _, _, _ = collate(rows, idx, pad_id, False)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                la, le = model(ids, att, first)
            la = F.log_softmax(la, -1).cpu().numpy().astype(np.float64)
            le = F.log_softmax(le, -1).cpu().numpy().astype(np.float64)
            for k, i in enumerate(idx):
                n = rows[i]["n"]
                if model.head == "bio":
                    out_a[i] = bio_to_span_probs(la[k, :n], ARG_TAGS, ARGUMENT_ROLES)
                    out_e[i] = bio_to_span_probs(le[k, :n], ENT_TAGS, ENTITY_TYPES)
                else:
                    out_a[i] = span_probs(la[k, :n])
                    out_e[i] = span_probs(le[k, :n])
    return out_a, out_e


def train_member(cfg, rows, pad_id, predict_sets):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = Tagger(cfg["backbone"], cfg["head"]).to(device)
    weight_a, weight_e = class_weights(rows, cfg["head"])
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    enc_decay, enc_plain, head = [], [], []
    for name, p in model.named_parameters():
        if not name.startswith("encoder."):
            head.append(p)
        elif "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            enc_plain.append(p)
        else:
            enc_decay.append(p)
    opt = torch.optim.AdamW([
        {"params": enc_decay, "lr": cfg["lr"], "weight_decay": WEIGHT_DECAY},
        {"params": enc_plain, "lr": cfg["lr"], "weight_decay": 0.0},
        {"params": head, "lr": HEAD_LR, "weight_decay": 0.0},
    ])
    base_lrs = [g["lr"] for g in opt.param_groups]
    order_rng = np.random.RandomState(cfg["seed"])
    total = math.ceil(len(rows) / BATCH_SIZE) * cfg["epochs"]
    warm = int(WARMUP * total)
    step = 0
    collected = [None] * len(predict_sets)
    taken = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = order_rng.permutation(len(rows))
        batches = []
        for s in range(0, len(perm), BATCH_SIZE * 8):
            chunk = sorted(perm[s: s + BATCH_SIZE * 8], key=lambda i: len(rows[i]["ids"]))
            batches += [chunk[c: c + BATCH_SIZE] for c in range(0, len(chunk), BATCH_SIZE)]
        order_rng.shuffle(batches)
        running = 0.0
        for idx in batches:
            ids, att, first, lengths, ya, ye, sa, se = collate(rows, idx, pad_id, True)
            f = step / max(1, warm) if step < warm else max(0.0, (total - step) / max(1, total - warm))
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * f
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                la, le = model(ids, att, first)
            loss = member_loss(model, la, le, lengths, ya, ye, sa, se, weight_a, weight_e)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            running += loss.item()
        print(f"  {cfg['backbone']} {cfg['head']} seed {cfg['seed']} epoch {epoch} loss {running / len(batches):.4f}", flush=True)
        if epoch >= cfg["epochs"] - SNAPSHOTS:
            taken += 1
            for j, rs in enumerate(predict_sets):
                pa, pe = predict(model, rs, pad_id)
                if collected[j] is None:
                    collected[j] = (pa, pe)
                else:
                    collected[j] = ([x + y for x, y in zip(collected[j][0], pa)], [x + y for x, y in zip(collected[j][1], pe)])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return [([x / taken for x in c[0]], [x / taken for x in c[1]]) for c in collected]


def select_spans(probs, key, labels, thresholds):
    n = probs.shape[0]
    label_probs = probs[:, :, 1:]
    best = label_probs.argmax(-1)
    best_p = np.take_along_axis(label_probs, best[:, :, None], -1)[:, :, 0]
    cand = []
    for s in range(n):
        for l in range(min(MAX_SPAN, n - s)):
            c = best[s, l]
            if best_p[s, l] > thresholds[c]:
                cand.append((best_p[s, l] - thresholds[c], s, l, c))
    cand.sort(key=lambda x: -x[0])
    used = np.zeros(n, dtype=bool)
    out = []
    for _, s, l, c in cand:
        if not used[s: s + l + 1].any():
            used[s: s + l + 1] = True
            out.append({key: labels[c], "start": int(s), "end": int(s + l + 1)})
    return sorted(out, key=lambda x: x["start"])


def span_f1(pred_docs, gold_docs, key, labels):
    scores = []
    for lab in labels:
        tp = n_pred = n_gold = 0
        for p, g in zip(pred_docs, gold_docs):
            ps = {(x["start"], x["end"]) for x in p if x[key] == lab}
            gs = {(x["start"], x["end"]) for x in g if x[key] == lab}
            tp += len(ps & gs)
            n_pred += len(ps)
            n_gold += len(gs)
        scores.append(2 * tp / (n_pred + n_gold) if n_pred + n_gold else 1.0)
    return float(np.mean(scores))


position = np.arange(len(train))
segment = len(train) / CALIBRATION_SEGMENTS
in_tail = (position % segment) >= segment * (1 - CALIBRATION_TAIL)
calib_fit = train[~in_tail].reset_index(drop=True)
calib_hold = train[in_tail].reset_index(drop=True)
hold_targets = [target_of[i] for i in calib_hold["id"]]
print("calibration fit", len(calib_fit), "hold", len(calib_hold), flush=True)

sum_a = None
sum_e = None
count = 0
hold_a = None
hold_e = None
for cfg in CALIBRATION_MEMBERS:
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    fit_rows = encode_rows(tok, calib_fit, True)
    hold_rows = encode_rows(tok, calib_hold, False)
    (ha, he), = train_member(cfg, fit_rows, tok.pad_token_id, [hold_rows])
    if hold_a is None:
        hold_a, hold_e = ha, he
    else:
        hold_a = [x + y for x, y in zip(hold_a, ha)]
        hold_e = [x + y for x, y in zip(hold_e, he)]

n_calib = len(CALIBRATION_MEMBERS)
hold_a = [x / n_calib for x in hold_a]
hold_e = [x / n_calib for x in hold_e]
attainable = 0.0
for thr in THRESHOLD_GRID:
    flat_a = np.full(len(ARGUMENT_ROLES), thr)
    flat_e = np.full(len(ENTITY_TYPES), thr)
    pred_a = [select_spans(a, "role", ARGUMENT_ROLES, flat_a) for a in hold_a]
    pred_e = [select_spans(e, "type", ENTITY_TYPES, flat_e) for e in hold_e]
    value = 0.5 * span_f1(pred_a, [t["arguments"] for t in hold_targets], "role", ARGUMENT_ROLES) + 0.5 * span_f1(pred_e, [t["entities"] for t in hold_targets], "type", ENTITY_TYPES)
    print(f"threshold {thr:.2f} holdout span F1 {value:.4f}", flush=True)
    attainable = max(attainable, value)
global_threshold = float(np.clip(attainable / 2.0, THRESHOLD_GRID[0], THRESHOLD_GRID[-1]))
arg_thresholds = np.full(len(ARGUMENT_ROLES), global_threshold)
ent_thresholds = np.full(len(ENTITY_TYPES), global_threshold)
print("attainable", round(attainable, 4), "selected threshold", round(global_threshold, 4), flush=True)


def write_submission(sum_a, sum_e, count):
    predictions = []
    for i, r in enumerate(test.itertuples(index=False)):
        brief = {"n_tokens": int(r.n_tokens), "arguments": select_spans(sum_a[i] / count, "role", ARGUMENT_ROLES, arg_thresholds), "entities": select_spans(sum_e[i] / count, "type", ENTITY_TYPES, ent_thresholds)}
        predictions.append(json.dumps(brief, separators=(",", ":")))
    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    n_args = sum(len(json.loads(p)["arguments"]) for p in predictions)
    n_ents = sum(len(json.loads(p)["entities"]) for p in predictions)
    print(f"wrote {len(submission)} rows from {count} members, {n_args} arguments, {n_ents} entities", flush=True)


train_rows_cache = {}
for cfg in MEMBERS:
    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    if cfg["backbone"] not in train_rows_cache:
        train_rows_cache[cfg["backbone"]] = (encode_rows(tok, train, True), encode_rows(tok, test, False))
    train_rows, test_rows = train_rows_cache[cfg["backbone"]]
    (ta, te), = train_member(cfg, train_rows, tok.pad_token_id, [test_rows])
    if sum_a is None:
        sum_a, sum_e = ta, te
    else:
        sum_a = [x + y for x, y in zip(sum_a, ta)]
        sum_e = [x + y for x, y in zip(sum_e, te)]
    count += 1
    write_submission(sum_a, sum_e, count)
