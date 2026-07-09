#!/bin/bash
set -e
cd /mnt/patient-unit/home/apoorv/repos/knowledge-cutoff
OUT=scratch_direct_report.txt
: > $OUT
for m in claude-fable-5 claude-opus-4-8 gpt-4o; do
  echo "############## $m / direct ##############" >> $OUT
  .venv/bin/kc eval --model $m --probe direct --concurrency 12 2>>/dev/null | grep -vE "it/s\]|it\]$" | tail -26 >> $OUT
  echo >> $OUT
done
echo "DONE" >> $OUT
