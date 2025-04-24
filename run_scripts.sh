#!/bin/bash

# Get the absolute path to the scripts directory
SCRIPT_DIR="$(cd "$(dirname "$0")/scripts" && pwd)"

# Change into the scripts directory
cd "$SCRIPT_DIR" || exit 1

# Run each script from within the scripts directory
for script in *.sh; do
    echo "Running $script..."
    bash "$script"
done
