#!/bin/bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for nodes in 1 2 4 8 16 32; do
    for replicate in 1 2 3; do
        "${root}/launch.sh" "${nodes}" "${replicate}"
    done
done
