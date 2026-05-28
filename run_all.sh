#!/bin/bash
# Fan the walk-forward x ablation x seed grid across the GPU (fixed beta).
set -u
PY=/home/zeus/miniconda3/bin/python
cd "$(dirname "$0")"
CONC=${1:-16}
mkdir -p results/cells
CMDS=$(mktemp)

emit() {   # $1 variant  $2 beta  $3 method
  for f in 0 1 2 3 4; do
    for s in 0 1; do
      echo "$PY src/train.py --variant $1 --fold $f --seed $s --beta $2 --method $3 --seq_len 126 --batch 16 --out results/cells/$1_f${f}_s${s}_b$2.json"
    done
  done
}

emit base  0     te >> "$CMDS"
emit vib   0.001 te >> "$CMDS"
emit prior 0     te >> "$CMDS"
emit full  0.001 te >> "$CMDS"

echo "launching $(wc -l < "$CMDS") cells, concurrency $CONC"
cat "$CMDS" | xargs -P "$CONC" -I CMD bash -c "CMD"
echo ALL_DONE
