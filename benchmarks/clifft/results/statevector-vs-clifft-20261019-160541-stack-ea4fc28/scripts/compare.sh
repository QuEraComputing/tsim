#!/bin/zsh
W=/Users/rafaelhaenel/Documents/quera/tsim/.claude/worktrees/clifft-benchmarks/benchmarks/clifft/comparison
NAMES=(distillation cultivation_d3 coherent_d3_r3 cultivation_d5 coherent_d5_r5 qv_q10 symft_msc_d7_postselected)
OUT=/tmp/sv_item3/final
mkdir -p $OUT; rm -f $OUT/*.jsonl
for tree in before after; do
  dir=/tmp/tsim-before
  [[ $tree == after ]] && dir=/Users/rafaelhaenel/Documents/quera/tsim/.claude/worktrees/tsim-perf
  for threads in all 1; do
    /tmp/sv_item3/wait_idle.sh | tail -1
    cd $dir
    if [[ $threads == 1 ]]; then
      NUMBA_NUM_THREADS=1 NPROC=1 caffeinate -i uv run --quiet python /tmp/sv_item3/bench_default.py $W $OUT/$tree-$threads.jsonl 6 $NAMES 2>&1 | grep -v Warn | sed "s/^/$tree $threads /"
    else
      caffeinate -i uv run --quiet python /tmp/sv_item3/bench_default.py $W $OUT/$tree-$threads.jsonl 6 $NAMES 2>&1 | grep -v Warn | sed "s/^/$tree $threads /"
    fi
  done
done
