cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" -gt 7000 ]; do sleep 60; done
echo "VRAM OK $(date +%H:%M:%S)"
conda run --no-capture-output -n max python -u solution.py . working/submission.csv
echo "RUN EXIT $? $(date +%H:%M:%S)"
