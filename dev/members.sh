cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag --snap 1 > dev/logs/$tag.log 2>&1; echo "END $tag $(date +%H:%M:%S) last=$(grep -E '^ep ' dev/logs/$tag.log | tail -1 | awk '{print $6}')"; }
run m_infoxlm_bio dev/tagger.py --model microsoft/infoxlm-large --epochs 10 --lr 2e-5
run m_mdeb_span dev/spanner.py --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --hd 150
run m_ibtl_span dev/spanner.py --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5 --hd 150
run m_infoxlm_span dev/spanner.py --model microsoft/infoxlm-large --epochs 10 --lr 2e-5 --hd 150
echo MEMBERSDONE
