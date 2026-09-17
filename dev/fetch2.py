import os, tarfile, shutil
from huggingface_hub import snapshot_download, list_repo_files
OUT="/content/eris_actionable/hfpack2"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
stage=OUT+"/hub"
for repo in ["microsoft/infoxlm-large"]:
    files=list_repo_files(repo); print(repo, files, flush=True)
    pats=["config.json","tokenizer_config.json","special_tokens_map.json","vocab.txt","tokenizer.json","spm.model","sentencepiece.bpe.model"]
    pats.append("model.safetensors" if "model.safetensors" in files else "pytorch_model.bin")
    p=snapshot_download(repo, allow_patterns=pats)
    d=os.path.join(stage,"models--"+repo.replace("/","--"),"snapshots","modelscope0000000000000000000000000000000")
    os.makedirs(d, exist_ok=True); os.makedirs(os.path.join(stage,"models--"+repo.replace("/","--"),"refs"), exist_ok=True)
    open(os.path.join(stage,"models--"+repo.replace("/","--"),"refs","main"),"w").write("modelscope0000000000000000000000000000000")
    for f in os.listdir(p):
        s=os.path.join(p,f)
        if os.path.isfile(s): shutil.copy(os.path.realpath(s), os.path.join(d,f)); print("  ",f,os.path.getsize(os.path.join(d,f)),flush=True)
with tarfile.open(OUT+"/models2.tar","w") as t: t.add(stage, arcname="hub")
print("tar", os.path.getsize(OUT+"/models2.tar"), flush=True)
shutil.rmtree(stage)
