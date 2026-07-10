#!/bin/bash
# Re-eval every model on both probes. Resumable: only events not already in
# each run file are queried, so this cheaply backfills newly-added events.
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
LOG=scratch_all.log; : > $LOG
MODELS="claude-fable-5 gpt-5.6-sol grok-4.5 gemini-3.5-flash gemini-3.1-pro claude-opus-4-8 claude-sonnet-5 gpt-4o"
for m in $MODELS; do
  for p in direct mcq; do
    echo "=== $m / $p @ $(date +%H:%M:%S) ===" >> $LOG
    .venv/bin/kc eval --model "$m" --probe "$p" --concurrency 12 >> $LOG 2>&1
  done
done
echo "ALL DONE" >> $LOG
