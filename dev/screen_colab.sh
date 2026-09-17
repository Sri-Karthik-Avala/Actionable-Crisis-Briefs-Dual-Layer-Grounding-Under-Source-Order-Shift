mkdir -p dev/logs dev/out
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; python -u dev/tagger.py --tag $tag "$@" > dev/logs/$tag.log 2>&1; rc=$?; echo "END $tag rc=$rc $(date +%H:%M:%S)"; grep -E '^ep ' dev/logs/$tag.log | tail -12; grep -E 'Error|error' dev/logs/$tag.log | tail -3; }
run xlmrl_crf0 --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --crf 0
run indobertweet_crf0 --model indolem/indobertweet-base-uncased --epochs 10 --lr 5e-5 --crf 0
run xlmrb_crf0 --model FacebookAI/xlm-roberta-base --epochs 10 --lr 5e-5 --crf 0
run indobertl_crf0 --model indobenchmark/indobert-large-p1 --epochs 10 --lr 2e-5 --crf 0
run mdeb_crf0 --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --crf 0
run twxlmrb_crf0 --model cardiffnlp/twitter-xlm-roberta-base --epochs 10 --lr 5e-5 --crf 0
run xlmrl_crf1 --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --crf 1
echo ALLDONE
