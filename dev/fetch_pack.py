import os, tarfile, shutil
from huggingface_hub import snapshot_download, list_repo_files
OUT="/content/eris_actionable/hfpack"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
stage=OUT+"/hub"
for repo in ["indolem/indobertweet-base-uncased","indobenchmark/indobert-large-p1"]:
    files=list_repo_files(repo)
    print(repo, files, flush=True)
    pats=["config.json","tokenizer_config.json","special_tokens_map.json","vocab.txt","tokenizer.json","spm.model","sentencepiece.bpe.model"]
    pats.append("model.safetensors" if "model.safetensors" in files else "pytorch_model.bin")
    p=snapshot_download(repo, allow_patterns=pats)
    dst=os.path.join(stage,"models--"+repo.replace("/","--"),"snapshots","modelscope0000000000000000000000000000000")
    os.makedirs(dst, exist_ok=True)
    os.makedirs(os.path.join(stage,"models--"+repo.replace("/","--"),"refs"), exist_ok=True)
    open(os.path.join(stage,"models--"+repo.replace("/","--"),"refs","main"),"w").write("modelscope0000000000000000000000000000000")
    for f in os.listdir(p):
        src=os.path.join(p,f)
        if os.path.isfile(src):
            shutil.copy(os.path.realpath(src), os.path.join(dst,f))
            print("  ", f, os.path.getsize(os.path.join(dst,f)), flush=True)
with tarfile.open(OUT+"/models.tar","w") as t:
    t.add(stage, arcname="hub")
print("tar", os.path.getsize(OUT+"/models.tar"), flush=True)
shutil.rmtree(stage)
