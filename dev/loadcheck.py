import sys
from transformers import AutoTokenizer, AutoModel
for m in sys.argv[1:]:
    t=AutoTokenizer.from_pretrained(m); mod=AutoModel.from_pretrained(m)
    print(m, type(t).__name__, t("banjir di jakarta", add_special_tokens=False)["input_ids"], t.cls_token_id, t.sep_token_id, t.pad_token_id, sum(p.numel() for p in mod.parameters())//1000000, "M", flush=True)
