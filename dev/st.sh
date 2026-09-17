for f in $(ls -tr dev/logs/*.log 2>/dev/null); do echo "== $f"; grep -E '^ep |^train |Error|Traceback|total ' $f | tail -12; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
tail -3 logs/screen1.log
