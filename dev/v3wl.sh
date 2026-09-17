cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 1 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
run v3_mdeb_span_wl1 dev/spanner.py --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --hd 150 --wloss 1.0
run v3_ibtl_span_wl1 dev/spanner.py --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5 --hd 150 --wloss 1.0
run v3_xlmrl_span_wl1 dev/spanner.py --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --hd 150 --wloss 1.0
echo V3WLDONE
