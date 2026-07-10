#!/bin/bash
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
LOG=scratch_gpt5x.log; : > $LOG
for m in gpt-5.5 gpt-5.4; do
  for p in direct mcq; do
    echo "=== $m / $p @ $(date +%H:%M:%S) ===" >> $LOG
    .venv/bin/kc eval --model "$m" --probe "$p" --concurrency 12 >> $LOG 2>&1
  done
done
echo "ALL DONE" >> $LOG
