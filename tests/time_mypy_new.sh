#!/usr/bin/env bash

# Enable recursive globbing
shopt -s globstar

# processes each file in the module individually
process_files() {
  # declare local variables
  local mod_path mod_name start_time end_time file file_name exec_time total_time

  # assign arguments to local variables
  mod_path=$1
  mod_name=$2
  start_time=$(date +%s.%N)

  # Run mypy on all Python files in the directory
  echo "$mod_name:"
  for file in "$mod_path"/**/*.{py,pyi}; do
    if [ -f "$file" ]; then
      file_name=$(basename "$file")
      exec_time=$( { time mypy "$file" >/dev/null 2>&1; } 2>&1 )
      printf "    %-36s %s s\n" "$file_name" "$exec_time"
    fi
  done

  # End the timer for the module
  end_time=$(date +%s.%N)
  total_time=$(echo "$end_time - $start_time" | bc)

  # Store the total time for the module
  module_times["$dir_name"]="$total_time"
}

# Process a whole module
process_module() {
  # declare local variables
  local mod_path mod_name exec_time
  # assign arguments to local variables
  mod_path=$1
  mod_name=$2

  # Run mypy on all Python files in the directory
  exec_time=$( { time mypy "$mod_path" >/dev/null 2>&1; } 2>&1 )
  printf "    %-32s %s s\n" "$mod_name" "$exec_time"

  # Store the total time for the module
  module_times["$mod_name"]="$exec_time"
}


process_directory() {
  # declare local variables
  local dir_path dir_name start_time end_time file file_name exec_time
  declare -A module_times  # Associative array to store total times per module

  # assign arguments to local variables
  dir_path=$1
  dir_name=$2
  start_time=$(date +%s.%N)

  # Run mypy on all modules in the directory
  echo "$dir_name:"
  for dir in "$dir_path"/*; do
    if [ -d "$dir" ]; then
      dir_name=$(basename "$dir")
      if [ "$FINE_GRAINED" = true ]; then
        process_files "$dir" "$dir_name"
      else
        process_module "$dir" "$dir_name"
      fi
    fi
  done
  # End the timer for the module
  end_time=$(date +%s.%N)

  # Store the total time for the module
  total_time=$(echo "$end_time - $start_time" | bc)

  # print the summary of module times
  make_summary
}


make_summary() {
  # Summary of module times, sorted in descending order
  echo "Summary of $dir_name:"
  for module in "${!module_times[@]}"; do
    printf "    %-24s %s s\n" "$module" "${module_times[$module]}"
  done | sort -k2 -rn

  # print the total time for the directory
  echo "Total execution time: $total_time seconds"
}

# Display help message
show_help() {
  echo "Usage: $0 [-h|--help] [-f|--fine-grained]"
  echo
  echo "Options:"
  echo "  -h, --help          Show this help message and exit"
  echo "  -f, --fine-grained  Enable fine-grained timing"
}

main() {
  # Parse command-line options
  FINE_GRAINED=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      -h|--help) show_help; exit 0 ;;
      -f|--fine-grained) FINE_GRAINED=true ;;
      *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
    shift
  done

  # Set the time format for the time command
  TIMEFORMAT='%R'
  # get project root with git
  PROJECT_ROOT=$(git rev-parse --show-toplevel)

  cd "$PROJECT_ROOT" || exit
  echo "Project root: $PROJECT_ROOT"

  # Run mypy on all directories in src/tsdm
  process_directory "$PROJECT_ROOT/src/tsdm" "src/tsdm"

  # Run mypy on all directories in tests
  process_directory "$PROJECT_ROOT/tests/tsdm" "Tests"
}

main "$@"
