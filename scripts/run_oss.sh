#!/bin/bash
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
LOG=scratch_oss.log; : > $LOG
for m in glm-5.2 deepseek-v4-pro; do
  for p in direct mcq; do
    echo "=== $m / $p @ $(date +%H:%M:%S) ===" >> $LOG
    .venv/bin/kc eval --model "$m" --probe "$p" --concurrency 10 >> $LOG 2>&1
  done
done
echo "ALL DONE" >> $LOG
