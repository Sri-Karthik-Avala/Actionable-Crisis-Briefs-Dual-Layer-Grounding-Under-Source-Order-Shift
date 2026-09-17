cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 3 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) $(grep -E '^ep ' dev/logs/$tag.log | awk '{print $6}' | sort -n | tail -1) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
run sw_span_base dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150
run sw_span_e18 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 18 --lr 5e-5 --hd 150
run sw_span_lr3 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 3e-5 --hd 150
run sw_span_wl1 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --wloss 1.0
run sw_span_ms12 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150 --maxspan 12
run sw_span_hd256 dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 256
run sw_bio_e18 dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 18 --lr 5e-5
run sw_bio_fl dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --subword firstlast
echo SWEEPDONE
