#!/bin/zsh
calm=0
while (( calm < 3 )); do
  cpu=$(ps -Ao pcpu= | awk '{s += $1} END {printf "%.0f", s}')
  if (( cpu < 300 )); then calm=$((calm + 1)); else calm=0; fi
  echo "$(date +%T) cpu ${cpu}% calm $calm/3"
  (( calm < 3 )) && sleep 60
done
