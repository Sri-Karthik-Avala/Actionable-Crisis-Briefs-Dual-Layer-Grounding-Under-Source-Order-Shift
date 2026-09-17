cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 1 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}') best=$(grep -E '^ep ' dev/logs/$tag.log | awk '{print $6}' | sort -n | tail -1)"; }
run v3_span_e10 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150
run v3_span_e16 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 16 --lr 5e-5 --hd 150
run v3_span_wl1 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --wloss 1.0
run v3_bio_e16 dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 16 --lr 5e-5
echo V3EXPDONE
