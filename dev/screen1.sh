cd /c/Users/srika/Downloads/eris_actionable
export USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY="conda run --no-capture-output -n max python dev/tagger.py"
mkdir -p dev/logs
run() { tag=$1; shift; echo "START $tag $(date +%H:%M:%S)"; $PY --tag $tag "$@" > dev/logs/$tag.log 2>&1; echo "END $tag rc=$? $(grep -E '^ep ' dev/logs/$tag.log | sort -k6 -n | tail -1)"; }
run xlmrb_crf0 --model xlm-roberta-base --epochs 10 --lr 5e-5 --crf 0
run xlmrb_crf1 --model xlm-roberta-base --epochs 10 --lr 5e-5 --crf 1
run mdeb_crf0 --model microsoft/mdeberta-v3-base --epochs 10 --lr 5e-5 --crf 0
run xlmrl_crf0 --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --crf 0
run xlmrb_fl --model xlm-roberta-base --epochs 10 --lr 5e-5 --crf 0 --subword firstlast
run xlmrb_s1 --model xlm-roberta-base --epochs 10 --lr 5e-5 --crf 0 --seed 1
run mdebnli_crf0 --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --epochs 10 --lr 5e-5 --crf 0
run xlmrl_crf1 --model FacebookAI/xlm-roberta-large --epochs 10 --lr 2e-5 --crf 1
echo ALLDONE
