cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 2 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
run sw_span_e18 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 18 --lr 5e-5 --hd 150
run sw_span_wl1 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --wloss 1.0
run lc_span_80 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --val_chunk 3
run lc_span_60 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --val_chunk 3 --exclude_chunk 4
echo SWEEP3DONE
