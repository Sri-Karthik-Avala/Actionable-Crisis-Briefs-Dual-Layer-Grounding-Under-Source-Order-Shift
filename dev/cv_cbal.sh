cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
M=indolem/indobertweet-base-uncased
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 1 > dev/logs/$tag.log 2>&1; echo "END $tag rc=$? $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
for c in 0 1 2 3 4; do
  run cv_ctrl_bio_c$c dev/tagger.py --model $M --epochs 10 --lr 5e-5 --val_chunk $c
  run cv_ctrl_span_c$c dev/spanner.py --model $M --epochs 10 --lr 5e-5 --hd 150 --val_chunk $c
  run cv_cb05_bio_c$c dev/tagger.py --model $M --epochs 10 --lr 5e-5 --val_chunk $c --cbal 0.5
  run cv_cb05_span_c$c dev/spanner.py --model $M --epochs 10 --lr 5e-5 --hd 150 --val_chunk $c --cbal 0.5
done
echo CVDONE
