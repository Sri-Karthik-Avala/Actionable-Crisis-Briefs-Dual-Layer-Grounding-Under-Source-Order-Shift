set -e
repo=$1; shift
hub=~/.cache/huggingface/hub/models--${repo/\//--}
snap=$hub/snapshots/modelscope0000000000000000000000000000000
mkdir -p $snap $hub/refs
printf "modelscope0000000000000000000000000000000" > $hub/refs/main
for f in "$@"; do
  if [ ! -s $snap/$f ]; then
    curl -sS -L --retry 5 -m 3000 -o $snap/$f "https://www.modelscope.cn/models/$repo/resolve/master/$f"
  fi
  echo "$f $(stat -c %s $snap/$f)"
done
