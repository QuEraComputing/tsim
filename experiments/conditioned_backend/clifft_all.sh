#!/bin/bash
# One clifft job at a time, each under a 20 GB address-space cap.
cd /home/rafaelha/quera/tsim-fuzz
run() { (ulimit -v 20000000; timeout 3000 .venv/bin/python /tmp/cb/clifft_ler.py "$@" 2>/dev/null | grep '^{'); }
run cultivation_d3 2e9 post
run coherent_d3_r1 5e7 decode
run coherent_d3_r3 5e7 decode
run coherent_d5_r1 5e7 decode
run distillation 2e8 dist
run tricycle_6 1e8 dist
run end2end_d5 1e8 maskdist
run tricycle_9 2e7 dist
run tricycle_12 1e7 dist
run cultivation_d7 2e7 dist
run cultivation_d5 2e9 post
echo DONE
