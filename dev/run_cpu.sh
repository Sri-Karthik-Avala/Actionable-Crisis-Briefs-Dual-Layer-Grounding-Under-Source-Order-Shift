cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
conda run --no-capture-output -n max python -u solution.py . working/submission_v2.csv
echo "RUN EXIT $? $(date +%H:%M:%S)"
