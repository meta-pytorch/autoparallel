import logging
from functools import cache
from typing import cast, NamedTuple, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
import dataclasses
import itertools

logger = logging.getLogger(__name__)


class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: list[int]


class DTensorRedistributePlanner:
    """
    This class is used to plan the collective calls to transform the local shard
    of the DTensor from its current spec to the target spec.

    Suppose there are N tensor dimensions and M mesh dimensions, the total
    possible state size will be (N+2)*M*M!.
    """

    _instances: dict = {}

    @dataclasses.dataclass(frozen=True)
    class DistState:
        placements: tuple[Placement, ...]
        device_order: tuple[int, ...]
        # logical_shape: tuple[int, ...]

        def __str__(self):
            return f"{self.placements}{self.device_order})"

        def __repr__(self):
            return self.__str__()

    @classmethod
    def _create_cache_key(cls, device_mesh, tensor_dimension):
        return (id(device_mesh), tensor_dimension)

    def __new__(cls, device_mesh, tensor_dimension):
        cache_key = cls._create_cache_key(device_mesh, tensor_dimension)

        if cache_key not in cls._instances:
            instance = super().__new__(cls)
            object.__setattr__(instance, "_cache_key", cache_key)

            instance._initialized = False
            cls._instances[cache_key] = instance

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        cls._instances.clear()

    def __init__(
        self,
        device_mesh,
        tensor_dimension: int,
    ) -> None:
        # Only initialize once
        if getattr(self, "_initialized", False):
            return
        self.device_mesh = device_mesh
        self.coordinate = device_mesh.get_coordinate()
        assert self.coordinate is not None
        self.tensor_dimension = tensor_dimension
        self.setup_collective_cost()
        self._initialized = True

    def setup_collective_cost(
        self,
        all_reduce_cost: int = 4,
        all_to_all_cost: int = 1,
        all_gather_cost: int = 2,
        chunk_cost: int = 0,
    ) -> None:
        """
        Set up the cost weights for different collective operations.

        Args:
            all_reduce_cost: Cost weight for all-reduce operations
            all_to_all_cost: Cost weight for all-to-all operations
            all_gather_cost: Cost weight for all-gather operations
        """
        # those can be turned in a handler considering the tensor dim size
        self.all_reduce_cost = all_reduce_cost
        self.all_to_all_cost = all_to_all_cost
        self.all_gather_cost = all_gather_cost
        self.chunk_cost = chunk_cost

    def generate_device_order_permutation(
        self, placements: tuple[Placement, ...], device_order: tuple[int, ...]
    ):
        """
        Generate all possible device order permutations for the given placements
        and device order without collective ops.

        Example:
            S(0)RR with device order [0, 1, 2] can be permuted to [0, 2, 1].
            This handles transition S(a)[x, y] -> S(a)[y, x] using the path:
                S(a)[x, y] -> S(a)[x]R[y] -> R[x,y] -> R[y,x] -> S(a)[y]R[x]
                -> S(a)[y, x]
        """

        def _generate_device_order_permutation_target_placement(
            placements: tuple[Placement, ...],
            device_order: tuple[int, ...],
            target_placement_type: type,
        ):
            target_order = []
            target_indices = []
            for idx, (p, order) in enumerate(zip(placements, device_order)):
                if isinstance(p, target_placement_type):
                    target_indices.append(idx)
                    target_order.append(order)
            # permute replicate_order and place back to device_order[replicate_indices]
            permutations = list(itertools.permutations(target_indices))
            permuted_device_orders = []
            for perm in permutations:
                new_device_order = list(device_order)
                for perm_idx, order in zip(perm, target_order):
                    new_device_order[perm_idx] = order
                permuted_device_orders.append(tuple(new_device_order))
            return permuted_device_orders

        ret = []
        for placement_type in [
            Replicate,
        ]:
            # do not permute Partial.
            ret.extend(
                _generate_device_order_permutation_target_placement(
                    placements, device_order, placement_type
                )
            )
        return ret

    def get_next_state(
        self, placements: tuple[Placement, ...], device_order: tuple[int, ...]
    ):
        # We map tensor dim to device mesh axis, similar to JAX way to represent
        # the sharding. Notation S(<tensor dim>)[<list of device dims>] means
        # <tensor dim> is sharded on <list of device dims>, where the <list of
        # device dims> is sorted by device order.

        # Blow are possible transition from one sharding state to another. We
        # use `S` for Shard, `R` for Replicate and `P` for Partial.
        # case 1. Shard(a) -> Shard(b), use all to all, apply to case:
        #   S(a)[x] -> S(b)[x] or
        #   S(a)[x,y]S(b)[z,k] -> S(a)[x]S(b)[z,k,y], where device order of `y``
        #   > device order of `z` and `k` (need confirm)

        # case 2. Shard() -> Replicate(), use all gather, apply to case:
        #   S(a)[x,y,z] -> S(a)[x,y]R[z]

        # case 3. Partial() -> Replicate(), use all reduce, apply to case:
        #   P[x,y] -> P[y]R[x] or P[x]R[y]

        # case 4. Replicate() -> Shard(), use chunk, apply to case:
        #   R[x,y]S(a)[z] -> R[x]S(a)[z,y] (`a` can be any tensor dim). Note that
        #   `y` must be after `z`.

        # list of [DistState, cost]
        all_next_state: dict[DTensorRedistributePlanner.DistState, int] = {}

        sorted_placements = sorted(
            enumerate(placements), key=lambda x: device_order[x[0]]
        )
        # map tensor dim to the last device mesh dim (based on device_order) and
        # the corresponding device order
        tensor_sharded_dim_to_last_mesh_dim = [
            [-1, -1] for _ in range(self.tensor_dimension)
        ]
        for order, (mesh_dim, p) in enumerate(sorted_placements):
            if isinstance(p, Shard):
                tensor_sharded_dim_to_last_mesh_dim[p.dim] = [order, mesh_dim]

        # handle case 1: Shard(a) -> Shard(b)
        # For S(a), S(b), only the last device order of S(a) and S(b) can be all to all interchangeably. (need confirm)
        for src_tensor_dim, (src_device_order, src_index) in enumerate(
            tensor_sharded_dim_to_last_mesh_dim
        ):
            for dst_tensor_dim, (dst_device_order, dst_index) in enumerate(
                tensor_sharded_dim_to_last_mesh_dim
            ):
                # try replace S(src_tensor_dim) with S(dst_tensor_dim) at src_index
                if src_device_order <= dst_device_order:
                    continue
                new_placements = list(placements)
                new_placements[src_index] = Shard(dst_tensor_dim)
                dist_state = self.DistState(tuple(new_placements), device_order)
                all_next_state[dist_state] = self.all_to_all_cost

        # handle case 2: Shard() -> Replicate()
        for tensor_dim, (order, mesh_dim) in enumerate(
            tensor_sharded_dim_to_last_mesh_dim
        ):
            # last dim of mesh_dims can be convert to Replicate with all reduce.
            if mesh_dim == -1:
                continue
            new_placement = list(placements)
            new_placement[mesh_dim] = Replicate()
            dist_state = self.DistState(tuple(new_placement), device_order)
            all_next_state[dist_state] = self.all_gather_cost

        # handle case 3: Partial() -> Replicate()
        for idx, (order, p) in enumerate(zip(device_order, placements)):
            if isinstance(p, Partial):
                new_placement = list(placements)
                new_placement[idx] = Replicate()
                dist_state = self.DistState(tuple(new_placement), device_order)
                all_next_state[dist_state] = self.all_reduce_cost

        # handle case 4: Replicate() -> Shard()
        for src_mesh_dim, (src_order, p) in enumerate(zip(device_order, placements)):
            if isinstance(p, Replicate):
                for tensor_dim, (target_order, target_mesh_dim) in enumerate(
                    tensor_sharded_dim_to_last_mesh_dim
                ):
                    if src_order > target_order:
                        new_placement = list(placements)
                        new_placement[src_mesh_dim] = Shard(tensor_dim)
                        dist_state = self.DistState(tuple(new_placement), device_order)
                        all_next_state[dist_state] = self.chunk_cost

        # expand with device order permutations
        expanded_all_next_state = all_next_state.copy()
        for dist_state, cost in all_next_state.items():
            for permuted_device_order in self.generate_device_order_permutation(
                dist_state.placements, dist_state.device_order
            ):
                if permuted_device_order == dist_state.device_order:
                    continue
                permuted_dist_state = self.DistState(
                    dist_state.placements, permuted_device_order
                )
                expanded_all_next_state[permuted_dist_state] = cost
        return expanded_all_next_state

    @cache  # noqa: B019
    def find_min_cost_path(
        self, src_state: DistState, dst_state: DistState
    ) -> list["DTensorRedistributePlanner.DistState"]:
        """
        Find the shortest path from src_state to dst_state using Dijkstra's
        algorithm.

        Args:
            src_state: The source state
            dst_state: The destination state

        Returns:
            A list of states representing the smallest cost path from src_state
            to dst_state
        """
        import heapq

        # Priority queue (cost, counter, state, path) for Dijkstra's algorithm
        # Use counter to break ties and avoid comparing DistState objects
        counter = 0
        pq: list[
            tuple[
                int,
                int,
                DTensorRedistributePlanner.DistState,
                list[DTensorRedistributePlanner.DistState],
            ]
        ] = [(0, counter, src_state, [src_state])]
        visited = set()
        while pq:
            cost, _, current_state, path = heapq.heappop(pq)
            if current_state == dst_state:
                return path
            if current_state in visited:
                continue
            visited.add(current_state)
            # Get all possible next states and their costs
            next_states = self.get_next_state(
                current_state.placements, current_state.device_order
            )
            for next_state, transition_cost in next_states.items():
                if next_state not in visited:
                    new_cost = cost + transition_cost
                    new_path = path + [next_state]
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, next_state, new_path))
        raise AssertionError(
            f"No path found from src_state {src_state} to dst_state {dst_state}"
        )

    def get_logical_shape(
        self,
        src_state: "DTensorRedistributePlanner.DistState",
        mesh_dim: int,
        full_tensor_shape: tuple[int, ...],
    ):
        # make src_placements[mesh_dim] = Replicate() and get the logical shape
        # after the change
        sorted_src_placements = sorted(
            enumerate(src_state.placements), key=lambda x: src_state.device_order[x[0]]
        )
        new_logical_shape = list(full_tensor_shape)
        for idx, src_placement in sorted_src_placements:
            if idx == mesh_dim:
                continue
            if isinstance(src_placement, Shard):
                new_size = src_placement._local_shard_size_and_offset(
                    new_logical_shape[src_placement.dim],
                    self.device_mesh.size(mesh_dim=idx),
                    self.coordinate[idx],
                )[0]
                new_logical_shape[src_placement.dim] = new_size
        return new_logical_shape

    def generate_optimal_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        full_tensor_shape: tuple[int, ...],
        src_device_order: Optional[tuple[int, ...]],
        dst_device_order: Optional[tuple[int, ...]],
    ) -> list[_TransformInfo]:
        if src_device_order is None:
            src_device_order = tuple(range(self.device_mesh.ndim))
        if dst_device_order is None:
            dst_device_order = tuple(range(self.device_mesh.ndim))
        src_state = self.DistState(src_spec.placements, src_device_order)
        dst_state = self.DistState(dst_spec.placements, dst_device_order)
        transform_infos: list[_TransformInfo] = []
        state_path = self.find_min_cost_path(src_state, dst_state)

        for cur_state, nxt_state in zip(state_path[:-1], state_path[1:]):
            # find the mesh_dim that is different between cur_state and nxt_state
            if cur_state.placements != nxt_state.placements:
                # skip the transition of device order permutation
                update_mesh_dim = -1
                for mesh_dim, (cur_placement, nxt_placement) in enumerate(
                    zip(cur_state.placements, nxt_state.placements)
                ):
                    if cur_placement != nxt_placement:
                        if update_mesh_dim != -1:
                            raise AssertionError(
                                "Multiple mesh_dims are different between cur_state and nxt_state"
                            )
                        update_mesh_dim = mesh_dim
                        logical_shape = self.get_logical_shape(
                            cur_state, mesh_dim, full_tensor_shape
                        )
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=update_mesh_dim,
                                src_dst_placements=(cur_placement, nxt_placement),
                                logical_shape=logical_shape,
                            )
                        )
        return transform_infos

    def generate_greedy_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
    ) -> list[_TransformInfo]:
        """
        Generate the transform infos from the source placements to the target placements.

        To transform from source to target placement it might have multiple steps, i.e. it
        might decompose Si -> Sj into Si -> R -> Sj.
        This would detect if there're mis-aligned/nested shardings between src/dst placements.
        E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
        in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
        the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
        the latter is the first sharding on tensor dimension 0.
        """
        # logical shape records the logic tensor shape on the mesh dimension
        # this is useful to ensure uneven sharding gets correct output shape
        initial_logical_shape = list(src_spec.shape)
        mesh_dims_to_logical_shape = [initial_logical_shape]
        transform_infos: list[_TransformInfo] = []
        if self.device_mesh.ndim == 1:
            # if device_mesh is 1D, redistribute is a simple direct transformation
            transform_infos.append(
                _TransformInfo(
                    mesh_dim=0,
                    src_dst_placements=(src_spec.placements[0], dst_spec.placements[0]),
                    logical_shape=initial_logical_shape,
                )
            )
            return transform_infos

        # Handle multi-dim device mesh placement redistribution
        # First, we need to build the logical shape for each mesh dim
        # for correct allgathering uneven shards on each mesh dim (with dynamic padding)
        for i, src in enumerate(src_spec.placements):
            current_logical_shape = mesh_dims_to_logical_shape[i]
            if isinstance(src, Shard):
                if i < self.device_mesh.ndim - 1:
                    # calculate and save the logical shape for this sharding
                    mesh_dim_size = self.device_mesh.size(mesh_dim=i)
                    local_shard_size, _ = src._local_shard_size_and_offset(
                        current_logical_shape[src.dim],
                        mesh_dim_size,
                        self.coordinate[i],
                    )
                    new_logical_shape = list(current_logical_shape)
                    new_logical_shape[src.dim] = local_shard_size
                    mesh_dims_to_logical_shape.append(new_logical_shape)
            else:
                mesh_dims_to_logical_shape.append(current_logical_shape)

        # Next, we need to derive the transform infos from src to dst placements,
        # here we use a greedy search with step by step state transformations
        current_placements = list(src_spec.placements)
        target_placements = list(dst_spec.placements)

        if src_spec.num_shards > 1:
            # If src_spec have sharding, it could potentially have sharding that is misaligned with dst_spec
            # a common case of this is nested sharding (i.e. (S(0), S(0)) -> (R, S(0))).
            # In those cases, we first traverse from inner placement to outer placement
            # to detect misaligned shardings and properly replicate nested sharding first.
            for mesh_dim in reversed(range(len(current_placements))):
                current = current_placements[mesh_dim]
                target = target_placements[mesh_dim]
                # If target is not Shard, we can directly redistribute since we are traversing from innner
                # to outer placements here
                if isinstance(target, Shard):
                    # If target is Shard, check for nested sharding on the tensor dim BEFORE the current mesh_dim
                    shard_dim = target.dim
                    current_mesh_sharding, target_mesh_sharding = [], []
                    for i, (s, p) in enumerate(
                        zip(current_placements, target_placements)
                    ):
                        if i >= mesh_dim:
                            break
                        if s.is_shard(shard_dim):
                            current_mesh_sharding.append(i)
                        if p.is_shard(shard_dim):
                            target_mesh_sharding.append(i)

                    if current_mesh_sharding != target_mesh_sharding:
                        # if current/target_placements have misaligned sharding on the tensor dim BEFORE the current
                        # mesh_dim, we need to replicate the tensor on the mesh dim first to clear the nested sharding
                        target = Replicate()

                if current != target:
                    transform_infos.append(
                        _TransformInfo(
                            mesh_dim=mesh_dim,
                            src_dst_placements=(current, target),
                            logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                        )
                    )
                    current_placements[mesh_dim] = target

        # We always traverse from outer placement to inner placement to collect the remaining
        # needed transform infos (i.e. the replication from nested sharding might need to further
        # perform resharding to Shard again)
        for mesh_dim, (current, target) in enumerate(
            zip(current_placements, target_placements)
        ):
            if current != target:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target
        return transform_infos


@cache
def _get_dtensor_redistribute_planner(
    device_mesh: DeviceMesh, tensor_dimension: int
) -> DTensorRedistributePlanner:
    """Factory function to create and cache DTensorRedistributePlanner instances."""
    return DTensorRedistributePlanner(device_mesh, tensor_dimension)


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    src_device_order: tuple[int, ...],
    dst_device_order: tuple[int, ...],
) -> list[_TransformInfo]:
    transform_infos: list[_TransformInfo] = []
    device_mesh = src_spec.device_mesh

    if src_device_order == range(src_spec.mesh.ndim) and dst_device_order == range(
        src_spec.mesh.ndim
    ):
        use_greedy_transform = True
    else:
        use_greedy_transform = False

    # Note: You can always try the optimal solution for all placements transformation.
    # I saw the run can pass but I am curious about the performance.
    # use_greedy_transform = False

    drp = _get_dtensor_redistribute_planner(device_mesh, len(src_spec.shape))
    if use_greedy_transform:
        transform_infos = drp.generate_greedy_transform_infos(src_spec, dst_spec)
    else:
        transform_infos = drp.generate_optimal_transform_infos(
            src_spec, dst_spec, src_spec.shape, src_device_order, dst_device_order
        )
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    src_device_order: tuple[int, ...],
    dst_device_order: tuple[int, ...],
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec, dst_spec, src_device_order, dst_device_order
    )


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    src_device_order: Optional[tuple[int, ...]] = None,
    dst_device_order: Optional[tuple[int, ...]] = None,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    if not src_device_order:
        src_device_order = tuple(range(current_spec.device_mesh.ndim))
    if not dst_device_order:
        dst_device_order = tuple(range(target_spec.device_mesh.ndim))

    if not isinstance(src_device_order, tuple):
        src_device_order = tuple(src_device_order)
    if not isinstance(dst_device_order, tuple):
        dst_device_order = tuple(dst_device_order)

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    has_symints = any(isinstance(s, torch.SymInt) for s in current_spec.shape) or any(
        isinstance(s, torch.SymInt) for s in target_spec.shape
    )
    if has_symints:
        transform_infos = _gen_transform_infos_non_cached(
            current_spec, target_spec, src_device_order, dst_device_order
        )
    else:
        transform_infos = _gen_transform_infos(
            current_spec, target_spec, src_device_order, dst_device_order
        )

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements
        device_mesh.size(mesh_dim=i)

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                new_local_tensor = target_placement._replicate_to_shard(
                    local_tensor, device_mesh, i, my_coordinate[i]
                )
            else:
                assert current.is_shard(), (
                    f"Current placement should be shard but found {current}"
                )
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    new_local_tensor = shard_spec._to_new_shard_dim(
                        local_tensor,
                        device_mesh,
                        i,
                        transform_info.logical_shape,
                        target_placement.dim,
                    )
        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        local_tensor = new_local_tensor

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor
