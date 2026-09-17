nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
ps aux | grep -E 'tagger|spanner|solution' | grep -v grep | awk '{print $2, $11, $12, $13, $14}'
tail -5 logs/sol_v2.log
