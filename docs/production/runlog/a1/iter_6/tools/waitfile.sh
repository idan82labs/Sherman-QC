#!/bin/zsh
# usage: waitfile.sh FILE PATTERN  -- wait until FILE contains PATTERN (max 40 min)
for i in {1..800}; do grep -q "$2" "$1" 2>/dev/null && exit 0; sleep 3; done; exit 1
