#!/usr/bin/env bash

start_time=$(date +%s.%N)  # Start the timer

# get project root with git
PROJECT_ROOT=$(git rev-parse --show-toplevel)
cd "$PROJECT_ROOT" || exit
echo "Project root: $PROJECT_ROOT"

# Run mypy on all directories in src/tsdm
echo "src/tsdm:"
for dir in "$PROJECT_ROOT/src/tsdm/"*; do
  name=$(basename "$dir")
  TIMEFORMAT='%R'
  exec_time=$( { time mypy "$dir" >/dev/null 2>&1; } 2>&1 )
  printf "%-24s %s\n" "$name" "$exec_time"
done

# Run mypy on all directories in tests
echo "Tests:"
for dir in "$PROJECT_ROOT/tests/tsdm/"*; do
  name=$(basename "$dir")
  TIMEFORMAT='%R'
  exec_time=$( { time mypy "$dir" >/dev/null 2>&1; } 2>&1 )
  printf "%-24s %s\n" "$name" "$exec_time"
done

end_time=$(date +%s.%N)  # End the timer
# Calculate total execution time
total_time=$(echo "$end_time - $start_time" | bc)

echo "Total execution time: $total_time seconds"
