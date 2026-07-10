#!/bin/bash
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
LOG=scratch_gempro.log; : > $LOG
for p in direct mcq; do
  echo "=== gemini-3.1-pro / $p @ $(date +%H:%M:%S) ===" >> $LOG
  .venv/bin/kc eval --model gemini-3.1-pro --probe "$p" --concurrency 12 >> $LOG 2>&1
done
echo "ALL DONE" >> $LOG
