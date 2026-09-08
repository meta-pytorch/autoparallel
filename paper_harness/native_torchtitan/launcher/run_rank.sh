#!/bin/bash

set -euo pipefail
exec /packages/torchtitan_additional_packages/runner/run_rank.sh "$@"
