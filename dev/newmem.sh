cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
echo "START m_mdeb_span $(date +%H:%M:%S)"
$PY dev/spanner.py --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --hd 150 --tag m_mdeb_span --snap 1 > dev/logs/m_mdeb_span.log 2>&1
echo "END m_mdeb_span $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/m_mdeb_span.log | tail -1 | awk '{print $6}')"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" -gt 11000 ]; do sleep 60; done
echo "VRAM FREE $(date +%H:%M:%S)"
$PY dev/spanner.py --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5 --hd 150 --tag m_ibtl_span --snap 1 > dev/logs/m_ibtl_span.log 2>&1
echo "END m_ibtl_span $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/m_ibtl_span.log | tail -1 | awk '{print $6}')"
echo NEWMEMDONE
