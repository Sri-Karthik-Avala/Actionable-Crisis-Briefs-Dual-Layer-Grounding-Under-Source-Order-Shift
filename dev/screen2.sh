mkdir -p dev/logs dev/out
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; python -u "$@" --tag $tag > dev/logs/$tag.log 2>&1; rc=$?; echo "END $tag rc=$rc $(date +%H:%M:%S)"; grep -E '^ep ' dev/logs/$tag.log | tail -12; grep -E 'Error' dev/logs/$tag.log | tail -3; }
run span_ibtw dev/spanner.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --hd 150
run span_xlmrl dev/spanner.py --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --hd 150
run ibtw_s1 dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --seed 1
run ibtw_lc75 dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --exclude_chunk 0
run ibtw_s2 dev/tagger.py --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --seed 2
run xlmrl_e15 dev/tagger.py --model FacebookAI/xlm-roberta-large --epochs 15 --lr 2e-5
echo ALLDONE
