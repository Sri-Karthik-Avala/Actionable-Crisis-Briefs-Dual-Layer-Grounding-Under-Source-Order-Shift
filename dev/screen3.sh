cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python -u"
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY "$@" --tag $tag > dev/logs/$tag.log 2>&1; echo "END $tag rc=$? $(date +%H:%M:%S) $(grep -E '^ep ' dev/logs/$tag.log | tail -1)"; }
run s3_ibtw_bio dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5
run s3_ibtw_span dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150
run s3_mdeb_span dev/spanner.py --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --hd 150
run s3_ibtl_span dev/spanner.py --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5 --hd 150
run s3_xlmrl_bio dev/tagger.py --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5
run s3_xlmrl_span dev/spanner.py --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --hd 150
run s3_mdeb_bio dev/tagger.py --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5
run s3_ibtl_bio dev/tagger.py --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5
echo ALLDONE3
