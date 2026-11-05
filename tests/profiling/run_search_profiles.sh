#!/usr/bin/env bash

set -u

if [[ $# -lt 2 ]]; then
    printf 'usage: %s [heuristic|lp] results-dir\n' "$0" >&2
    exit 2
fi

mode=$1
results=$2
python=${PYTHON:-python}
revision=$(git rev-parse --short HEAD)
runner=tests/search_profile.py
failed=0

run_profile() {
    label=$1
    timeout_value=$2
    shift 2
    output="$results/$label"
    mkdir -p "$output"
    command=(
        env PYTHONPATH=. PYTHONHASHSEED=0 "$python" "$runner" "$@"
        --revision-label "$revision" --detailed-solution
        --output "$output/result.json"
    )
    printf '%q ' "${command[@]}" >"$output/command.txt"
    printf '\n' >>"$output/command.txt"
    if [[ -n "$timeout_value" ]]; then
        timed_command=(timeout --signal=TERM --kill-after=30s "$timeout_value")
    else
        timed_command=()
    fi
    /usr/bin/time -v -o "$output/time.txt" \
        "${timed_command[@]}" "${command[@]}" \
        >"$output/stdout.log" 2>"$output/stderr.log"
    status=$?
    printf '%s\n' "$status" >"$output/status.txt"
    if [[ $status -ne 0 ]]; then
        failed=1
    fi
}

if [[ "$mode" == heuristic ]]; then
    run_profile llama1b_full 20m \
        --model llama1b --mesh 2,4,8 --solver approx --lazy-costs true
    run_profile llama1b_split 20m \
        --model llama1b --mesh 2,4,8 --solver approx --lazy-costs true --seeded
    run_profile llama8b_full 20m \
        --model llama8b --mesh 2,4,8 --solver approx --lazy-costs true
    run_profile llama8b_split 20m \
        --model llama8b --mesh 2,4,8 --solver approx --lazy-costs true --seeded
    run_profile dsv3_3d_full 20m \
        --model dsv3 --moe-layout 3d --solver approx --lazy-costs true
    run_profile dsv3_3d_split 20m \
        --model dsv3 --moe-layout 3d --solver approx --lazy-costs true --seeded
    run_profile dsv3_4d_split 20m \
        --model dsv3 --moe-layout 4d --solver approx --lazy-costs true --seeded
elif [[ "$mode" == lp ]]; then
    run_profile llama1b_3d_lp "" \
        --model llama1b --mesh 2,4,8 --solver lp
    run_profile llama8b_3d_lp "" \
        --model llama8b --mesh 2,4,8 --solver lp
    run_profile dsv3_3d_lp "" \
        --model dsv3 --moe-layout 3d --solver lp
else
    printf 'usage: %s [heuristic|lp] results-dir\n' "$0" >&2
    exit 2
fi

exit "$failed"
