#!/bin/bash
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
LOG=scratch_frontier.log; : > $LOG
for m in gpt-5.6-sol grok-4.5 gemini-3.5-flash claude-sonnet-5; do
  for p in direct mcq; do
    echo "=== $m / $p @ $(date +%H:%M:%S) ===" >> $LOG
    .venv/bin/kc eval --model "$m" --probe "$p" --concurrency 12 >> $LOG 2>&1
    echo "--- done $m/$p ---" >> $LOG
  done
done
echo "ALL DONE" >> $LOG
