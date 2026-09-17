cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 1 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
run v3_span_cb03 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --cbal 0.3
run v3_span_cb05 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --cbal 0.5
run v3_span_cb03_s1 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --cbal 0.3 --seed 1
run v3_span_e10_s1 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --seed 1
echo V3CBDONE
