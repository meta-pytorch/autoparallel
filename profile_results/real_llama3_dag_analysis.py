import csv
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

sys.path.insert(0, "/home/wangkj/workspace/torchtitan")

from torchtitan.models.llama3 import llama3_configs  # noqa: E402

from autoparallel.api import AutoParallel
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config


WORLD_SIZE = 64
SEQ_LEN = 256
GLOBAL_BATCH = 64


def init_dist():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "fake", store=FakeStore(), rank=0, world_size=WORLD_SIZE
        )


def target_name(node):
    target = node.target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def layer_id(node):
    stacks = []
    for key in ("nn_module_stack", "fwd_nn_module_stack"):
        value = node.meta.get(key)
        if value:
            stacks.append(str(value))
    text = " ".join(stacks)
    match = re.search(r"layers[._']+([0-9]+)", text)
    return int(match.group(1)) if match else ""


def phase(node):
    if "fwd_nn_module_stack" in node.meta:
        return "backward"
    if "nn_module_stack" in node.meta:
        return "forward"
    if node.op == "placeholder" and str(node.name).startswith("tangents"):
        return "backward"
    return "unknown"


def bitset_counts(nodes, edges):
    idx = {node: i for i, node in enumerate(nodes)}
    children = [[] for _ in nodes]
    parents = [[] for _ in nodes]
    for src, dst in edges:
        children[idx[src]].append(idx[dst])
        parents[idx[dst]].append(idx[src])

    descendants = [0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        bits = 0
        for child in children[i]:
            bits |= 1 << child
            bits |= descendants[child]
        descendants[i] = bits

    ancestors = [0] * len(nodes)
    for i in range(len(nodes)):
        bits = 0
        for parent in parents[i]:
            bits |= 1 << parent
            bits |= ancestors[parent]
        ancestors[i] = bits

    return (
        [bits.bit_count() for bits in ancestors],
        [bits.bit_count() for bits in descendants],
    )


def treewidth_upper_bounds(edges):
    graph = nx.Graph()
    graph.add_edges_from(edges)
    width_min_fill, _ = nx.approximation.treewidth_min_fill_in(graph)
    width_min_degree, _ = nx.approximation.treewidth_min_degree(graph)

    moral = graph.copy()
    parents_by_child = defaultdict(list)
    for src, dst in edges:
        parents_by_child[dst].append(src)
    for parents in parents_by_child.values():
        for i, left in enumerate(parents):
            for right in parents[i + 1 :]:
                moral.add_edge(left, right)
    moral_width_min_fill, _ = nx.approximation.treewidth_min_fill_in(moral)
    moral_width_min_degree, _ = nx.approximation.treewidth_min_degree(moral)
    return {
        "undirected_min_fill": width_min_fill,
        "undirected_min_degree": width_min_degree,
        "moralized_min_fill": moral_width_min_fill,
        "moralized_min_degree": moral_width_min_degree,
        "undirected_edges": graph.number_of_edges(),
        "moralized_edges": moral.number_of_edges(),
    }


def run_analysis(out_dir):
    init_dist()
    mesh = torch.distributed.device_mesh.init_device_mesh(
        "cuda", (64,), mesh_dim_names=("dp",)
    )
    set_nccl_topo_config(detect_nccl_topo_config(mesh))

    config = llama3_configs["3B"](attn_backend="sdpa")
    config.rope.max_seq_len = SEQ_LEN
    with torch.device("meta"):
        model = config.build()

    def input_fn():
        return torch.randint(0, config.vocab_size, (GLOBAL_BATCH, SEQ_LEN), device="cuda")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    t0 = time.perf_counter()
    with AutoParallel(
        model, input_fn, mesh, mp_policy, repeated_subgraphs=True
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([(Shard(0),)])
        autop.add_output_constraints([(Shard(0),)])
        opt = autop.sharding_optimizer

        ilp_nodes = [node for node in opt.nodes if node.op != "output"]
        ilp_node_set = set(ilp_nodes)
        edges = []
        dep_args = {}
        dep_unique = {}
        for node in ilp_nodes:
            inputs = [inp for inp in opt._all_input_nodes(node) if inp in ilp_node_set]
            dep_args[node] = len(inputs)
            dep_unique[node] = len(set(inputs))
            for inp in set(inputs):
                edges.append((inp, node))

        offspring = Counter()
        for src, _dst in edges:
            offspring[src] += 1

        ancestor_counts, descendant_counts = bitset_counts(ilp_nodes, edges)
        node_to_idx = {node: i for i, node in enumerate(ilp_nodes)}
        treewidth = treewidth_upper_bounds(edges)

        rows = []
        for node in ilp_nodes:
            idx = node_to_idx[node]
            rows.append(
                {
                    "idx": idx,
                    "name": node.name,
                    "op": node.op,
                    "target": target_name(node),
                    "phase": phase(node),
                    "layer": layer_id(node),
                    "direct_dependency_args": dep_args[node],
                    "direct_dependency_nodes": dep_unique[node],
                    "direct_offspring_nodes": offspring[node],
                    "ancestor_count": ancestor_counts[idx],
                    "descendant_count": descendant_counts[idx],
                    "strategy_count": len(opt.strats[node].strategies),
                }
            )

        merge_points = [
            row for row in rows if int(row["direct_dependency_nodes"]) > 1
        ]
        merge_points.sort(
            key=lambda row: (
                -int(row["direct_dependency_nodes"]),
                -int(row["descendant_count"]),
                int(row["idx"]),
            )
        )
        fanout_points = sorted(
            rows,
            key=lambda row: (-int(row["direct_offspring_nodes"]), int(row["idx"])),
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_csv = out_dir / "real_llama3_3b_dag_node_stats.csv"
    with node_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    merge_csv = out_dir / "real_llama3_3b_merge_points.csv"
    with merge_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(merge_points)

    summary = {
        "model": "LLaMA3 3B",
        "mesh": "1D 64",
        "trace_and_optimizer_build_s": time.perf_counter() - t0,
        "ilp_nodes": len(ilp_nodes),
        "dag_edges": len(edges),
        "merge_points": len(merge_points),
        "branch_points": sum(1 for row in rows if int(row["direct_offspring_nodes"]) > 1),
        "max_direct_dependency_nodes": max(int(row["direct_dependency_nodes"]) for row in rows),
        "max_direct_offspring_nodes": max(int(row["direct_offspring_nodes"]) for row in rows),
        "max_ancestor_count": max(int(row["ancestor_count"]) for row in rows),
        "max_descendant_count": max(int(row["descendant_count"]) for row in rows),
        "treewidth_upper_bounds": treewidth,
        "direct_dependency_histogram": dict(
            sorted(Counter(int(row["direct_dependency_nodes"]) for row in rows).items())
        ),
        "direct_offspring_histogram": dict(
            sorted(Counter(int(row["direct_offspring_nodes"]) for row in rows).items())
        ),
        "top_merge_points": merge_points[:30],
        "top_fanout_points": fanout_points[:30],
        "node_stats_csv": str(node_csv),
        "merge_points_csv": str(merge_csv),
    }
    summary_path = out_dir / "real_llama3_3b_dag_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )
    run_analysis("profile_results")


if __name__ == "__main__":
    main()
