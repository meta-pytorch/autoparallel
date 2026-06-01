"""Diagnose the bare approx gap: is the factor graph FAITHFUL (scores the true
optimum correctly -> solver is at fault) or UNFAITHFUL (drops cost -> model is at
fault), and is the optimum REPRESENTABLE in the group choices (pruning)?

Builds the ILP, solves it exactly with CBC, then checks whether the approx's own
machinery (total_objective + factor graph) reproduces the CBC optimum, and where
the approx's own solution differs. Env: MODEL, MESH, SEQLEN."""
import logging
import os
import time
from collections import defaultdict
from unittest.mock import patch

import pulp
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from autoparallel._testing.models.llama3 import Transformer, TransformerModelArgs
from autoparallel.api import AutoParallel
from autoparallel.approximate_sharding import ApproximateShardingSolver
from autoparallel.cost_models.collective_runtime_estimation import set_nccl_topo_config
from autoparallel.cost_models.nccl_cost_model import detect_nccl_topo_config

logging.basicConfig(level=logging.ERROR)
for fn, val in [("device_count", lambda: 8), ("get_device_name", lambda *a, **k: "H100"),
                ("get_device_capability", lambda *a, **k: (9, 0))]:
    patch(f"torch.cuda.{fn}", val).start()
patch("torch.cuda.get_device_properties", lambda *a, **k: type(
    "P", (), {"major": 9, "minor": 0, "name": "H100",
              "total_memory": 80 * 1024**3, "multi_processor_count": 132})()).start()

MODEL = os.environ.get("MODEL", "1b")
SEQLEN = int(os.environ.get("SEQLEN", "2048"))
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MESH", "8,8").split(","))
ws = 1
for d in MESH_SHAPE:
    ws *= d
names = {2: ("dp", "tp"), 3: ("dp", "cp", "tp")}[len(MESH_SHAPE)]
torch.distributed.init_process_group("fake", store=FakeStore(), rank=0, world_size=ws)
mesh = torch.distributed.device_mesh.init_device_mesh("cuda", MESH_SHAPE, mesh_dim_names=names)
ndim = mesh.ndim
vocab_size = 128256
batch_size = 2 * mesh.shape[0]
_CFG = {
    "1b": dict(dim=2048, n_layers=16, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.5, multiple_of=256),
    "3b": dict(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8, ffn_dim_multiplier=1.0, multiple_of=256),
    "8b": dict(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=1024),
    "70b": dict(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8, ffn_dim_multiplier=1.3, multiple_of=4096),
}


def model_fn():
    args = TransformerModelArgs(rope_theta=500000, vocab_size=vocab_size,
                                max_seq_len=SEQLEN, **_CFG[MODEL])
    with torch.device("meta"):
        return Transformer(args)


def input_fn():
    return torch.randint(0, vocab_size, (batch_size, SEQLEN), device="cuda")


def constrain(autop):
    x = (Shard(0),) + (Replicate(),) * (ndim - 1)
    out = (Shard(0), Shard(2)) if ndim == 2 else x
    autop.add_parameter_memory_constraint(low=None, high=None)
    autop.add_input_constraints([x])
    autop.add_output_constraints([out])


set_nccl_topo_config(detect_nccl_topo_config(mesh))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
print(f"### diag MODEL={MODEL} mesh={MESH_SHAPE}{names} ###", flush=True)

t = time.perf_counter()
autop = AutoParallel(model_fn(), input_fn, mesh, mp, repeated_subgraphs=True, solver="ilp")
autop.__enter__()
constrain(autop)
opt = autop.sharding_optimizer
print(f"[build] {time.perf_counter()-t:.1f}s decision_vars={len(opt.decision_vars)}", flush=True)

opt._set_objective()
opt._apply_memory_constraint()
t = time.perf_counter()
opt.prob.solve(pulp.PULP_CBC_CMD(msg=False, options=["preprocess off"]))
obj_cbc = pulp.value(opt.prob.objective)
print(f"[cbc] solve={time.perf_counter()-t:.1f}s obj={obj_cbc:.1f} status={pulp.LpStatus[opt.prob.status]}", flush=True)

# CBC per-(root)node chosen out_idx
cbc_out = {}
for key, var in opt.pulp_variables.items():
    v = var.varValue
    if v is not None and v > 0.5:
        cbc_out[key[0]] = key[2]

approx = ApproximateShardingSolver(opt)
approx._build_problem()
approx._build_factors()

# (A) FAITHFULNESS: exact objective of the CBC solution via the approx machinery.
approx.cur_out = dict(cbc_out)
e_cbc_total = approx.total_objective()
print(f"[faithful] approx.total_objective(CBC soln) = {e_cbc_total:.1f}  "
      f"(CBC obj {obj_cbc:.1f}; match={abs(e_cbc_total-obj_cbc)<1.0})", flush=True)

# (B) REPRESENTABILITY: can the group choices express the CBC solution?
cbc_full = dict(cbc_out)
for copy_idx, root_idx in opt.cluster_links.items():
    if root_idx in cbc_out:
        cbc_full[copy_idx] = cbc_out[root_idx]
unrep = []
cbc_group_choice = {}
for gid, g in enumerate(approx.groups):
    found = None
    for ci, choice in enumerate(g.choices):
        if all(cbc_full.get(m) == o for m, o in choice.items()):
            found = ci
            break
    if found is None:
        unrep.append(gid)
    else:
        cbc_group_choice[gid] = found
print(f"[representable] groups={len(approx.groups)} "
      f"with_no_matching_choice={len(unrep)}", flush=True)

# (C) factor-graph energy of the CBC solution (if representable)
if not unrep:
    for gid, ci in cbc_group_choice.items():
        approx._set_group(gid, ci)
    fge = approx._fast_total_energy()
    print(f"[fg-energy] _fast_total_energy(CBC soln) = {fge:.1f} "
          f"(match CBC {abs(fge-obj_cbc)<1.0})", flush=True)

# (D) run the normal approx, localize where it differs from CBC
approx2 = ApproximateShardingSolver(opt)
approx2.get_solution(verbose=False)
obj_approx = opt.profile["approximate"]["objective"]
ax_out = dict(approx2.cur_out)
print(f"[approx] obj={obj_approx:.1f} gap={100*(obj_approx-obj_cbc)/obj_cbc:+.2f}%", flush=True)

# per-node exact cost under each assignment (cost_bearing nodes), to localize gap
def node_cost(solver, out_map, v):
    o = out_map[v]
    node = opt.nodes[v]
    strat = opt.strats[node].strategies[o]
    prod = solver._arg_prod.get(v, {})
    c = 0.0
    for argi in range(len(strat.redistribute_cost)):
        p = prod.get(argi)
        inp = out_map[p] if (p is not None and p in out_map) else 0
        key = (v, argi, o, inp)
        dv = opt.decision_vars.get(key)
        if dv is None:
            return None
        c += dv.cost
    return solver.node_mult[v] * c

diffs = []
for v in approx2.cost_bearing:
    if cbc_out.get(v) != ax_out.get(v):
        c_cbc = node_cost(approx2, cbc_out, v)
        c_ax = node_cost(approx2, ax_out, v)
        if c_cbc is not None and c_ax is not None:
            diffs.append((c_ax - c_cbc, v, opt.nodes[v].name, cbc_out.get(v), ax_out.get(v)))
diffs.sort(reverse=True)
print(f"[localize] {len(diffs)} cost-bearing nodes differ; top contributors (approx-cbc):", flush=True)
for d, v, name, oc, oa in diffs[:15]:
    print(f"    +{d:10.1f}  node={name[:40]:40s} cbc_out={oc} approx_out={oa}", flush=True)
tot = sum(d for d, *_ in diffs)
print(f"[localize] total node-cost diff over differing nodes = {tot:.1f} "
      f"(gap = {obj_approx-obj_cbc:.1f})", flush=True)
