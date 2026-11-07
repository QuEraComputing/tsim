#!/bin/bash
cd /home/rafaelha/quera/tsim-cond-backend
run() { for st in active_statevector conditioned_statevector; do seed=$([ $st = active_statevector ] && echo 101 || echo 202); .venv/bin/python /tmp/cb/ler.py $1 $st $2 $3 $seed 2>/dev/null | grep '^{'; done; }
run cultivation_d3 2e9 post
run cultivation_d5 4e9 post
run coherent_d3_r1 5e7 decode
run coherent_d3_r3 5e7 decode
run coherent_d5_r1 5e7 decode
run end2end_d5 2e8 maskdist
run cultivation_d7 2e8 dist
run distillation 2e8 dist
run tricycle_6 1e8 dist
run tricycle_9 1e8 dist
echo DONE
