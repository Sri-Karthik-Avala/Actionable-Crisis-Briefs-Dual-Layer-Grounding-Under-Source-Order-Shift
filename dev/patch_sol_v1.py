p="solution.py"
s=open(p,encoding="utf-8").read()
rep=[
('''MEMBERS = [
    {"backbone": "FacebookAI/xlm-roberta-large", "lr": 2e-5, "epochs": 8, "seed": 11},
]''',
'''MEMBERS = [
    {"backbone": "FacebookAI/xlm-roberta-large", "lr": 2e-5, "epochs": 10, "seed": 11, "crf": True},
    {"backbone": "indobenchmark/indobert-large-p1", "lr": 2e-5, "epochs": 10, "seed": 12, "crf": False},
    {"backbone": "microsoft/mdeberta-v3-base", "lr": 5e-5, "epochs": 10, "seed": 13, "crf": False},
    {"backbone": "FacebookAI/xlm-roberta-large", "lr": 2e-5, "epochs": 10, "seed": 14, "crf": True},
    {"backbone": "indolem/indobertweet-base-uncased", "lr": 5e-5, "epochs": 10, "seed": 15, "crf": False},
    {"backbone": "FacebookAI/xlm-roberta-large", "lr": 2e-5, "epochs": 10, "seed": 16, "crf": True},
]'''),
('amp_dtype = torch.bfloat16\n', 'amp_dtype = torch.float16\n'),
('''class Tagger(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone).float()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.arg_head = nn.Linear(hidden, len(ARG_TAGS))
        self.ent_head = nn.Linear(hidden, len(ENT_TAGS))
''',
'''class CRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        trans, start = transition_masks(tags)
        self.register_buffer("trans_mask", torch.tensor(np.maximum(trans, -1e4), dtype=torch.float32))
        self.register_buffer("start_mask", torch.tensor(np.maximum(start, -1e4), dtype=torch.float32))
        self.trans = nn.Parameter(torch.zeros(len(tags), len(tags)))
        self.start = nn.Parameter(torch.zeros(len(tags)))

    def nll(self, em, y, mask):
        trans = self.trans + self.trans_mask
        start = self.start + self.start_mask
        m = mask.float()
        gold = start[y[:, 0]] + em[:, 0].gather(1, y[:, :1]).squeeze(1)
        alpha = start[None] + em[:, 0]
        for t in range(1, em.size(1)):
            step_gold = em[:, t].gather(1, y[:, t:t + 1]).squeeze(1) + trans[y[:, t - 1], y[:, t]]
            gold = gold + step_gold * m[:, t]
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None] + em[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        return (torch.logsumexp(alpha, 1) - gold).mean() / mask.float().sum(1).mean()


class Tagger(nn.Module):
    def __init__(self, backbone, use_crf):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone).float()
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.arg_head = nn.Linear(hidden, len(ARG_TAGS))
        self.ent_head = nn.Linear(hidden, len(ENT_TAGS))
        self.use_crf = use_crf
        if use_crf:
            self.arg_crf = CRF(ARG_TAGS)
            self.ent_crf = CRF(ENT_TAGS)

    def loss(self, la, le, ya, ye, wm):
        if self.use_crf:
            return self.arg_crf.nll(la, ya, wm) + self.ent_crf.nll(le, ye, wm)
        return F.cross_entropy(la[wm], ya[wm]) + F.cross_entropy(le[wm], ye[wm])
'''),
('    model = Tagger(cfg["backbone"]).to(device)\n', '    model = Tagger(cfg["backbone"], cfg["crf"]).to(device)\n    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")\n'),
('''            loss = F.cross_entropy(la[wm], ya[wm]) + F.cross_entropy(le[wm], ye[wm])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()''',
'''            loss = model.loss(la, le, ya, ye, wm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()'''),
]
for a,b in rep:
    assert s.count(a)==1, a[:80]
    s=s.replace(a,b)
open(p,"w",encoding="utf-8",newline="\n").write(s)
print("patched")
