cd /root/.cache/huggingface/hub
for m in models--indolem--indobertweet-base-uncased models--indobenchmark--indobert-large-p1; do for s in $m/snapshots/*/; do echo $s; ls -laL $s | awk '{print $5, $9}'; done; done
