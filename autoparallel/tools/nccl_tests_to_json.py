# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""Convert nccl-tests outputs into a MeasuredCommTable JSON file.

Each input is the stdout of one all_gather_perf, reduce_scatter_perf,
all_reduce_perf or alltoall_perf run without NCCL_TESTS_SPLIT_MASK. The
collective, node count and ranks per node are read from its header.

Example:
    python -m autoparallel.tools.nccl_tests_to_json results/*.txt \\
        --gpus-per-node 8 --provenance nccl_version=2.27.5 -o h100.json

Use the table with
    h100_topo_config(..., measured_table=MeasuredCommTable.load("h100.json"))
"""

import argparse

from autoparallel.cost_models.measured_comm_table import MeasuredCommTable


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="nccl-tests output files")
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument(
        "--provenance",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="metadata recorded in the table (repeatable)",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    provenance = dict(item.split("=", 1) for item in args.provenance)
    table = MeasuredCommTable.from_nccl_tests(
        args.inputs, args.gpus_per_node, provenance
    )
    table.save(args.output)
    print(f"Wrote {len(table.curves)} curves to {args.output}")


if __name__ == "__main__":
    main()
