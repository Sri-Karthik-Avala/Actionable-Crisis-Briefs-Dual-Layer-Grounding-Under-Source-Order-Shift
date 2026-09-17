cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
conda run --no-capture-output -n max python -u solution.py . working/submission.csv
echo "RUN EXIT $? $(date +%H:%M:%S)"
