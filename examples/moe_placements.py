from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.distributed.tensor._collective_utils import fill_empty_tensor_to_shards


class DynamicShard(Placement):
    """
    DynamicShard allows variable-length chunks along the specified tensor dimension.

    Unlike the standard Shard placement which assumes equal-sized chunks,
    DynamicShard handles variable chunk sizes across a mesh dimension.
    """

    def __init__(self, dim: int):
        """
        DynamicShard allows variable-length chunks along the specified tensor dimension.

        Args:
            dim (int): The tensor dimension to shard.
        """
        self.dim = dim

    def _compute_offsets(
        self, chunk_sizes: List[Union[int, torch.SymInt]]
    ) -> List[Union[int, torch.SymInt]]:
        """
        Compute offsets from chunk sizes using cumulative sum.
        Returns offsets including the final endpoint.
        """
        offsets: List[Union[int, torch.SymInt]] = [0]
        current_offset = 0
        for size in chunk_sizes:
            current_offset += size
            offsets.append(current_offset)
        return offsets

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        *,
        with_padding: bool = False,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Split tensor into variable-sized chunks based on provided chunk_sizes.

        Args:
            tensor: Tensor to split
            chunk_sizes: List of chunk sizes for splitting the tensor
            with_padding: Ignored - DynamicShard never pads to maintain variable sizes
            contiguous: Whether to make tensors contiguous

        Returns:
            Tuple of (chunk_list, pad_sizes) where pad_sizes is always [0, 0, ...]
        """
        assert (
            self.dim < tensor.ndim
        ), f"Sharding dim {self.dim} greater than or equal to tensor ndim {tensor.ndim}"

        # Validate chunk_sizes parameter
        if not isinstance(chunk_sizes, (list, tuple)):
            raise TypeError(
                "chunk_sizes must be a list or tuple of ints or torch.SymInt"
            )
        if len(chunk_sizes) == 0:
            raise ValueError("chunk_sizes must not be empty")

        # Validate chunk size types and values
        for i, size in enumerate(chunk_sizes):
            if not (isinstance(size, int) or isinstance(size, torch.SymInt)):
                raise TypeError(
                    f"Chunk size at index {i} must be an int or torch.SymInt, got {type(size)}"
                )
            if isinstance(size, int) and size < 0:
                raise ValueError(
                    f"Chunk size at index {i} must be non-negative, got {size}"
                )

        # Get the actual tensor size along the sharding dimension
        tensor_dim_size = tensor.size(self.dim)

        # Check that sum of chunk sizes matches tensor dimension length
        total_chunk_size = sum(chunk_sizes)
        if all(isinstance(size, int) for size in chunk_sizes):
            # For all int chunk_sizes, use assertion
            assert total_chunk_size == tensor_dim_size, (
                f"Sum of chunk sizes ({total_chunk_size}) must equal tensor dimension size ({tensor_dim_size}) "
                f"on dim {self.dim}. chunk_sizes: {chunk_sizes}"
            )
        else:
            # For SymInt chunk_sizes, use torch._check
            torch._check(
                total_chunk_size == tensor_dim_size,
                lambda: f"Sum of chunk sizes ({total_chunk_size}) must equal tensor dimension size ({tensor_dim_size}) "
                f"on dim {self.dim}. chunk_sizes: {chunk_sizes}",
            )

        # Create chunks based on chunk sizes - unified logic for int and SymInt
        tensor_list: List[torch.Tensor] = []
        current_offset = 0

        for chunk_size in chunk_sizes:
            if chunk_size == 0:
                # Create empty chunk
                chunk_shape = list(tensor.shape)
                chunk_shape[self.dim] = 0
                chunk = tensor.new_empty(chunk_shape)
            else:
                # Use narrow for both int and SymInt (PyTorch handles both)
                chunk = tensor.narrow(self.dim, current_offset, chunk_size)

            if contiguous:
                chunk = chunk.contiguous()

            tensor_list.append(chunk)
            current_offset += chunk_size

        # Fill empty tensors if needed (compatibility)
        tensor_list = fill_empty_tensor_to_shards(
            tensor_list, self.dim, len(chunk_sizes) - len(tensor_list)
        )

        # DynamicShard never pads - return zero pad sizes
        pad_sizes = [0] * len(tensor_list)
        return tensor_list, pad_sizes

    def _chunk_slices(
        self, tensor_shape: Sequence[int], chunk_sizes: List[Union[int, torch.SymInt]]
    ):
        """
        Returns a list of slice objects for each chunk along the sharded dimension,
        based on the chunk sizes.

        Args:
            tensor_shape (Sequence[int]): The shape of the tensor to be sharded.
            chunk_sizes (List[Union[int, torch.SymInt]]): The list of chunk sizes.

        Returns:
            List[tuple]: List of slice tuples for each chunk.
        """
        if self.dim >= len(tensor_shape):
            raise ValueError(
                f"Sharding dim {self.dim} >= tensor ndim {len(tensor_shape)}"
            )

        # Compute offsets from chunk sizes
        offsets = self._compute_offsets(chunk_sizes)

        slices = []
        for i in range(len(chunk_sizes)):
            start = offsets[i]
            end = offsets[i + 1]

            # Validate bounds for concrete values
            if isinstance(start, int) and isinstance(end, int):
                if start < 0 or end > tensor_shape[self.dim]:
                    raise ValueError(
                        f"Slice [{start}:{end}] out of bounds for dimension size {tensor_shape[self.dim]}"
                    )

            slc = [slice(None)] * len(tensor_shape)
            slc[self.dim] = slice(start, end)
            slices.append(tuple(slc))
        return slices

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension with variable chunk sizes.
        Uses torch.distributed.scatter which can handle variable-sized tensors.

        Args:
            tensor: Tensor to shard and scatter
            chunk_sizes: List of chunk sizes for each rank in the mesh dimension
            mesh: DeviceMesh for distribution
            mesh_dim: Dimension of the mesh to scatter across
            src_data_rank: Source rank for scattering (default: 0)

        Returns:
            Local chunk of the tensor for this rank
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # if rank is not part of mesh, return empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # Skip communications, just split locally
            scatter_list, _ = self._split_tensor(
                tensor, chunk_sizes, with_padding=False, contiguous=True
            )
            return scatter_list[mesh_dim_local_rank]

        # Get the process group for this mesh dimension
        group = mesh.get_group(mesh_dim)

        if mesh_dim_local_rank == src_data_rank:
            # Source rank: split tensor and create scatter_list
            scatter_list, _ = self._split_tensor(
                tensor, chunk_sizes, with_padding=False, contiguous=True
            )
        else:
            # Non-source ranks: scatter_list should be None
            scatter_list = None

        # Create output tensor for this rank's chunk
        chunk_size = chunk_sizes[mesh_dim_local_rank]
        if chunk_size == 0:
            output_shape = list(tensor.shape)
            output_shape[self.dim] = 0
            output = tensor.new_empty(output_shape, requires_grad=tensor.requires_grad)
        else:
            output_shape = list(tensor.shape)
            output_shape[self.dim] = chunk_size
            output = tensor.new_empty(output_shape, requires_grad=tensor.requires_grad)

        # Use torch.distributed.scatter which handles variable-sized tensors
        # The src parameter expects rank within the process group, not global rank
        torch.distributed.scatter(output, scatter_list, src=src_data_rank, group=group)

        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Reduce and scatter with variable chunk sizes.
        Cannot use reduce_scatter_tensor since chunks are not equal-sized.

        Args:
            tensor: Tensor to reduce and scatter
            chunk_sizes: List of chunk sizes for each rank in the mesh dimension
            mesh: DeviceMesh for distribution
            reduce_op: Reduction operation ('sum', 'avg', etc.)
            mesh_dim: Dimension of the mesh to reduce across

        Returns:
            Local reduced chunk for this rank
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            return tensor

        # For DynamicShard, we implement reduce_scatter manually:
        # 1. All-reduce the full tensor
        # 2. Each rank locally extracts their chunk

        # Step 1: All-reduce
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        reduced_tensor = funcol.all_reduce(
            tensor, reduceOp=reduce_op, group=(mesh, mesh_dim)
        )

        # Step 2: Each rank extracts their own chunk
        mesh_dim_local_rank = my_coordinate[mesh_dim]
        chunk_size = chunk_sizes[mesh_dim_local_rank]

        if chunk_size == 0:
            # Return empty tensor
            chunk_shape = list(tensor.shape)
            chunk_shape[self.dim] = 0
            return tensor.new_empty(chunk_shape, requires_grad=tensor.requires_grad)

        # Compute offset for this rank
        offset = sum(chunk_sizes[:mesh_dim_local_rank])
        result = reduced_tensor.narrow(self.dim, offset, chunk_size)

        return result.contiguous()

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        All-gather variable-sized shards to reconstruct the full replicated tensor.
        Uses torch.distributed.all_gather which supports uneven sized tensors.

        Args:
            local_tensor: Local tensor shard
            chunk_sizes: List of chunk sizes for all ranks in the mesh dimension
            mesh: DeviceMesh for distribution
            mesh_dim: Dimension of the mesh to gather across

        Returns:
            Full reconstructed tensor
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # Create empty tensor with the logical shape
            total_size = sum(chunk_sizes)
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        # Get the process group for this mesh dimension
        group = mesh.get_group(mesh_dim)

        # Make sure local tensor is contiguous
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        # Prepare tensor list for all_gather - we don't pre-allocate since
        # all_gather with uneven tensors will handle the allocation
        gathered_tensors = []
        torch.distributed.all_gather(gathered_tensors, local_tensor, group=group)

        # Filter out empty chunks and concatenate
        valid_chunks = [chunk for chunk in gathered_tensors if chunk.size(self.dim) > 0]

        if not valid_chunks:
            # All chunks are empty
            total_size = sum(chunk_sizes)
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        result = torch.cat(valid_chunks, dim=self.dim)
        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Transform from replicated tensor to a sharded tensor by taking the appropriate chunk.

        Args:
            local_tensor: Full replicated tensor
            chunk_sizes: List of chunk sizes for all ranks
            mesh: DeviceMesh for distribution
            mesh_dim: Dimension of the mesh
            shard_index: Index of the shard to extract

        Returns:
            Local shard for the specified index
        """
        shards, _ = self._split_tensor(
            local_tensor,
            chunk_sizes,
            with_padding=False,
            contiguous=False,
        )
        return shards[shard_index].clone()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicShard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash(self.dim)

    def __repr__(self) -> str:
        """Machine readable representation of the DynamicShard placement"""
        return f"DynamicShard(dim={self.dim})"

    def __str__(self) -> str:
        """Human readable representation of the DynamicShard placement"""
        return f"DS({self.dim})"


@dataclass(frozen=True)
class PartitionedShard(Placement):
    """
    The PartitionedShard placement describes a DTensor that is sharded on a tensor dimension
    where each shard contains multiple partitions of potentially variable sizes.

    This placement type is particularly useful for MoE (Mixture of Experts) models where
    different experts may have different sizes, and we need to maintain partition alignment
    across different sharding strategies.

    Args:
        dim (int): The tensor dimension that describes how the DTensor is sharded
        num_partitions (int): Total number of partitions across all shards
        splits (List[Union[int, torch.SymInt]]): Number of elements in each partition
        aligned (bool): Whether partitions are aligned across shards or not
    """

    dim: int
    num_partitions: int
    splits: List[Union[int, torch.SymInt]]
    aligned: bool = False

    def __post_init__(self):
        """Validate the PartitionedShard configuration."""
        if self.dim < 0:
            raise ValueError(f"Sharding dimension must be non-negative, got {self.dim}")

        if self.num_partitions <= 0:
            raise ValueError(
                f"Number of partitions must be positive, got {self.num_partitions}"
            )

        if len(self.splits) != self.num_partitions:
            raise ValueError(
                f"Number of splits ({len(self.splits)}) must equal num_partitions ({self.num_partitions})"
            )

        # Validate split sizes
        for i, split in enumerate(self.splits):
            if isinstance(split, int) and split < 0:
                raise ValueError(
                    f"Split at index {i} must be non-negative, got {split}"
                )

    def _compute_offsets(self) -> List[Union[int, torch.SymInt]]:
        """
        Compute cumulative offsets from partition splits.
        Returns offsets including the final endpoint.
        """
        offsets: List[Union[int, torch.SymInt]] = [0]
        current_offset = 0
        for size in self.splits:
            current_offset += size
            offsets.append(current_offset)
        return offsets

    def _split_tensor_by_partitions(
        self,
        tensor: torch.Tensor,
        *,
        contiguous: bool = True,
    ) -> List[torch.Tensor]:
        """
        Split tensor into partitions based on self.splits.

        Args:
            tensor: Tensor to split into partitions
            contiguous: Whether to make tensors contiguous

        Returns:
            List of partition tensors
        """
        assert (
            self.dim < tensor.ndim
        ), f"Sharding dim {self.dim} >= tensor ndim {tensor.ndim}"

        # Validate that sum of splits matches tensor dimension
        total_split_size = sum(self.splits)
        tensor_dim_size = tensor.size(self.dim)

        if all(isinstance(size, int) for size in self.splits):
            assert total_split_size == tensor_dim_size, (
                f"Sum of splits ({total_split_size}) must equal tensor dimension size "
                f"({tensor_dim_size}) on dim {self.dim}"
            )
        else:
            torch._check(
                total_split_size == tensor_dim_size,
                lambda: f"Sum of splits ({total_split_size}) must equal tensor dimension size "
                f"({tensor_dim_size}) on dim {self.dim}",
            )

        # Create partition chunks
        partition_list: List[torch.Tensor] = []
        current_offset = 0

        for split_size in self.splits:
            if split_size == 0:
                # Create empty partition
                chunk_shape = list(tensor.shape)
                chunk_shape[self.dim] = 0
                partition = tensor.new_empty(chunk_shape)
            else:
                partition = tensor.narrow(self.dim, current_offset, split_size)

            if contiguous:
                partition = partition.contiguous()

            partition_list.append(partition)
            current_offset += split_size

        return partition_list

    def _compute_aligned_splits(
        self, mesh_size: int, rank: int
    ) -> List[Union[int, torch.SymInt]]:
        """
        Compute the partition splits for a specific rank in aligned mode.

        Args:
            mesh_size: Size of the mesh dimension
            rank: Rank within the mesh dimension

        Returns:
            List of partition sizes for this rank
        """
        if self.num_partitions % mesh_size != 0:
            raise ValueError(
                f"For aligned partitioning, num_partitions ({self.num_partitions}) "
                f"must be divisible by mesh_size ({mesh_size})"
            )

        partitions_per_shard = self.num_partitions // mesh_size
        start_partition = rank * partitions_per_shard
        end_partition = start_partition + partitions_per_shard

        return self.splits[start_partition:end_partition]

    def _compute_unaligned_splits(
        self, mesh_size: int, rank: int
    ) -> List[Union[int, torch.SymInt]]:
        """
        Compute the partition splits for a specific rank in unaligned mode.
        Each rank gets a slice of every partition.

        Args:
            mesh_size: Size of the mesh dimension
            rank: Rank within the mesh dimension

        Returns:
            List of partition slice sizes for this rank
        """
        unaligned_splits = []

        for partition_size in self.splits:
            # Compute slice size for this rank using torch.chunk semantics
            if isinstance(partition_size, int):
                full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                start_idx = min(rank * full_chunk_size, partition_size)
                end_idx = min(start_idx + full_chunk_size, partition_size)
                slice_size = end_idx - start_idx
            else:
                # SymInt case - use symbolic computation
                full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                start_idx = torch.sym_min(rank * full_chunk_size, partition_size)
                end_idx = torch.sym_min(start_idx + full_chunk_size, partition_size)
                slice_size = end_idx - start_idx

            unaligned_splits.append(slice_size)

        return unaligned_splits

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        Convert partitioned shard to replicated tensor.
        Routes to appropriate method based on alignment.
        """
        if self.aligned:
            return self._aligned_to_replicate(
                local_tensor, mesh, mesh_dim, current_logical_shape
            )
        else:
            return self._unaligned_to_replicate(
                local_tensor, mesh, mesh_dim, current_logical_shape
            )

    def _aligned_to_replicate(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        Convert aligned partitioned shard to replicated tensor.

        Process:
        1. All-gather with list of tensors (handles dynamic sizes)
        2. Concatenate to form complete tensor
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # Create empty tensor with logical shape
            total_size = sum(self.splits)
            result_shape = list(current_logical_shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        mesh_size = mesh.size(mesh_dim)

        # Make local tensor contiguous
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        # All-gather from all ranks
        gathered_tensors = []
        torch.distributed.all_gather(
            gathered_tensors, local_tensor, group=mesh.get_group(mesh_dim)
        )

        # For aligned sharding, we need to reorder gathered tensors
        # to match the original partition order
        partitions_per_shard = self.num_partitions // mesh_size

        # Split each gathered tensor into its constituent partitions
        all_partitions = []
        for rank, gathered_tensor in enumerate(gathered_tensors):
            rank_splits = self._compute_aligned_splits(mesh_size, rank)
            rank_partitions = []
            current_offset = 0

            for split_size in rank_splits:
                if split_size == 0:
                    chunk_shape = list(gathered_tensor.shape)
                    chunk_shape[self.dim] = 0
                    partition = gathered_tensor.new_empty(chunk_shape)
                else:
                    partition = gathered_tensor.narrow(
                        self.dim, current_offset, split_size
                    )

                rank_partitions.append(partition)
                current_offset += split_size

            all_partitions.extend(rank_partitions)

        # Reorder partitions to original order and concatenate
        ordered_partitions = []
        for partition_idx in range(self.num_partitions):
            # Find which rank and local index this partition belongs to
            rank = partition_idx // partitions_per_shard
            local_idx = partition_idx % partitions_per_shard
            global_idx = rank * partitions_per_shard + local_idx
            ordered_partitions.append(all_partitions[global_idx])

        # Filter out empty partitions and concatenate
        valid_partitions = [p for p in ordered_partitions if p.size(self.dim) > 0]

        if not valid_partitions:
            # All partitions are empty
            total_size = sum(self.splits)
            result_shape = list(current_logical_shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(valid_partitions, dim=self.dim)

    def _unaligned_to_replicate(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        Convert unaligned partitioned shard to replicated tensor.

        Process:
        1. All-gather to collect all shards
        2. Perform partition alignment using splits
        3. Return fully reconstructed tensor
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            total_size = sum(self.splits)
            result_shape = list(current_logical_shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        mesh_size = mesh.size(mesh_dim)

        # Make local tensor contiguous
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        # All-gather from all ranks
        gathered_tensors = []
        torch.distributed.all_gather(
            gathered_tensors, local_tensor, group=mesh.get_group(mesh_dim)
        )

        # For unaligned sharding, each rank has slices of all partitions
        # We need to reconstruct each partition by combining slices from all ranks
        reconstructed_partitions = []

        for partition_idx in range(self.num_partitions):
            partition_slices = []

            # Collect slice of this partition from each rank
            for rank, gathered_tensor in enumerate(gathered_tensors):
                rank_splits = self._compute_unaligned_splits(mesh_size, rank)

                # Find offset to this partition's slice in the rank's tensor
                offset = sum(rank_splits[:partition_idx])
                slice_size = rank_splits[partition_idx]

                if slice_size == 0:
                    # Empty slice - skip
                    continue

                partition_slice = gathered_tensor.narrow(self.dim, offset, slice_size)
                partition_slices.append(partition_slice)

            # Concatenate slices to form complete partition
            if partition_slices:
                complete_partition = torch.cat(partition_slices, dim=self.dim)
                reconstructed_partitions.append(complete_partition)

        # Concatenate all partitions to form final tensor
        if not reconstructed_partitions:
            total_size = sum(self.splits)
            result_shape = list(current_logical_shape)
            result_shape[self.dim] = total_size
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(reconstructed_partitions, dim=self.dim)

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Transform from replicated tensor to partitioned shard.
        Routes to appropriate method based on alignment.
        """
        if self.aligned:
            return self._replicate_to_aligned_shard(
                local_tensor, mesh, mesh_dim, shard_index
            )
        else:
            return self._replicate_to_unaligned_shard(
                local_tensor, mesh, mesh_dim, shard_index
            )

    def _replicate_to_unaligned_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Convert replicated tensor to unaligned partitioned shard.

        Requirements:
        - num_partitions: Total number of partitions
        - splits: Partition sizes within each shard
        - Chunk size: Sum of splits (uniform across shards except possibly last)
        """
        mesh_size = mesh.size(mesh_dim)

        # First split the tensor into partitions
        partitions = self._split_tensor_by_partitions(local_tensor, contiguous=False)

        # Get the splits for this rank
        rank_splits = self._compute_unaligned_splits(mesh_size, shard_index)

        # Extract slices from each partition for this rank
        rank_slices = []
        for partition_idx, partition in enumerate(partitions):
            slice_size = rank_splits[partition_idx]

            if slice_size == 0:
                # Empty slice
                continue

            # Compute offset for this rank's slice within the partition
            partition_size = self.splits[partition_idx]
            if isinstance(partition_size, int):
                full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                start_idx = min(shard_index * full_chunk_size, partition_size)
            else:
                # SymInt case
                full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                start_idx = torch.sym_min(shard_index * full_chunk_size, partition_size)

            partition_slice = partition.narrow(self.dim, start_idx, slice_size)
            rank_slices.append(partition_slice)

        # Concatenate slices to form this rank's unaligned shard
        if not rank_slices:
            # All slices are empty
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(rank_slices, dim=self.dim).clone()

    def _replicate_to_aligned_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Convert replicated tensor to aligned partitioned shard.

        Requirements:
        - num_partitions: Total number of partitions
        - partitions_per_shard: num_partitions / mesh_size
        - Variable partition sizes allowed
        - Fixed number of partitions per shard
        """
        mesh_size = mesh.size(mesh_dim)

        if self.num_partitions % mesh_size != 0:
            raise ValueError(
                f"For aligned partitioning, num_partitions ({self.num_partitions}) "
                f"must be divisible by mesh_size ({mesh_size})"
            )

        # Split tensor into all partitions
        partitions = self._split_tensor_by_partitions(local_tensor, contiguous=False)

        # Get partition range for this rank
        partitions_per_shard = self.num_partitions // mesh_size
        start_partition = shard_index * partitions_per_shard
        end_partition = start_partition + partitions_per_shard

        # Extract partitions for this rank
        rank_partitions = partitions[start_partition:end_partition]

        # Filter out empty partitions and concatenate
        valid_partitions = [p for p in rank_partitions if p.size(self.dim) > 0]

        if not valid_partitions:
            # All partitions are empty
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(valid_partitions, dim=self.dim).clone()

    def _unaligned_to_aligned_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Convert unaligned partitioned shard to aligned partitioned shard.

        Algorithm:
        1. Calculate partitions per shard: num_partitions_per_shard = num_partitions / mesh_size
        2. First all-to-all: Exchange split information
        3. Compute boundaries for data exchange
        4. Second all-to-all: Exchange tensor data using boundaries
        5. Local reordering using out_splits to achieve final alignment
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # Return empty tensor
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        mesh_size = mesh.size(mesh_dim)
        my_rank = my_coordinate[mesh_dim]

        if self.num_partitions % mesh_size != 0:
            raise ValueError(
                f"For aligned partitioning, num_partitions ({self.num_partitions}) "
                f"must be divisible by mesh_size ({mesh_size})"
            )

        partitions_per_shard = self.num_partitions // mesh_size

        # Step 1: Get current unaligned splits for this rank
        current_splits = self._compute_unaligned_splits(mesh_size, my_rank)

        # Step 2: First all-to-all - exchange split information
        # We need to gather splits from all ranks and reorganize them for aligned layout
        all_splits = []
        torch.distributed.all_gather_object(
            all_splits, current_splits, group=mesh.get_group(mesh_dim)
        )

        # Reorganize splits: from [rank][partition] to [partition][rank] layout
        target_splits = []
        for partition_idx in range(self.num_partitions):
            # Find which shard this partition should go to in aligned layout
            target_rank = partition_idx // partitions_per_shard
            # Find position within that shard
            local_partition_idx = partition_idx % partitions_per_shard

            # If this is my target rank, collect the split
            if target_rank == my_rank:
                # Collect splits from all source ranks for this partition
                partition_splits = [
                    rank_splits[partition_idx] for rank_splits in all_splits
                ]
                target_splits.extend(partition_splits)

        # Step 3: Compute boundaries for data exchange
        # in_boundaries: current chunk sizes (what we have)
        in_boundaries = current_splits

        # out_boundaries: target chunk sizes (what we want to receive)
        out_boundaries = target_splits

        # Step 4: Second all-to-all - exchange tensor data
        # Split local tensor according to current partitions
        tensor_chunks = []
        current_offset = 0

        for split_size in in_boundaries:
            if split_size == 0:
                chunk_shape = list(local_tensor.shape)
                chunk_shape[self.dim] = 0
                chunk = local_tensor.new_empty(chunk_shape)
            else:
                chunk = local_tensor.narrow(self.dim, current_offset, split_size)

            tensor_chunks.append(chunk.contiguous())
            current_offset += split_size

        # Prepare all-to-all scatter list for sending data
        scatter_list = []
        for target_rank in range(mesh_size):
            # Collect chunks that should go to target_rank
            rank_chunks = []
            for partition_idx in range(self.num_partitions):
                if partition_idx // partitions_per_shard == target_rank:
                    rank_chunks.append(tensor_chunks[partition_idx])

            if rank_chunks:
                concatenated = torch.cat(rank_chunks, dim=self.dim)
            else:
                # Send empty tensor
                empty_shape = list(local_tensor.shape)
                empty_shape[self.dim] = 0
                concatenated = local_tensor.new_empty(empty_shape)

            scatter_list.append(concatenated)

        # Perform all-to-all
        gathered_chunks = []
        torch.distributed.all_to_all(
            gathered_chunks, scatter_list, group=mesh.get_group(mesh_dim)
        )

        # Step 5: Local reordering - concatenate received chunks in aligned order
        valid_chunks = [chunk for chunk in gathered_chunks if chunk.size(self.dim) > 0]

        if not valid_chunks:
            # All chunks are empty
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(valid_chunks, dim=self.dim)

    def _aligned_to_unaligned_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Convert aligned partitioned shard to unaligned partitioned shard.
        This performs the reverse operation of unaligned_to_aligned_shard.
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # Return empty tensor
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        mesh_size = mesh.size(mesh_dim)
        my_rank = my_coordinate[mesh_dim]

        if self.num_partitions % mesh_size != 0:
            raise ValueError(
                f"For aligned partitioning, num_partitions ({self.num_partitions}) "
                f"must be divisible by mesh_size ({mesh_size})"
            )

        partitions_per_shard = self.num_partitions // mesh_size

        # Step 1: Get current aligned splits for this rank
        current_splits = self._compute_aligned_splits(mesh_size, my_rank)

        # Step 2: First all-to-all - exchange split information
        # Need to transpose from [shard][partition_within_shard] to [partition][shard]
        all_splits = []
        torch.distributed.all_gather_object(
            all_splits, current_splits, group=mesh.get_group(mesh_dim)
        )

        # Compute target unaligned splits for this rank
        target_splits = self._compute_unaligned_splits(mesh_size, my_rank)

        # Step 3: Prepare data for all-to-all exchange
        # Split local tensor according to current aligned partitions
        tensor_chunks = []
        current_offset = 0

        for split_size in current_splits:
            if split_size == 0:
                chunk_shape = list(local_tensor.shape)
                chunk_shape[self.dim] = 0
                chunk = local_tensor.new_empty(chunk_shape)
            else:
                chunk = local_tensor.narrow(self.dim, current_offset, split_size)

            tensor_chunks.append(chunk.contiguous())
            current_offset += split_size

        # Step 4: Compute which partitions to send to which ranks
        scatter_list = []

        for target_rank in range(mesh_size):
            # For each target rank, collect the slices of partitions they need
            rank_chunks = []

            # For each partition this rank currently has
            for local_partition_idx, chunk in enumerate(tensor_chunks):
                # Find the global partition index
                global_partition_idx = (
                    my_rank * partitions_per_shard + local_partition_idx
                )

                # Compute how much of this partition goes to target_rank
                partition_size = self.splits[global_partition_idx]

                if isinstance(partition_size, int):
                    full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                    start_idx = min(target_rank * full_chunk_size, partition_size)
                    end_idx = min(start_idx + full_chunk_size, partition_size)
                    slice_size = end_idx - start_idx
                else:
                    # SymInt case
                    full_chunk_size = (partition_size + mesh_size - 1) // mesh_size
                    start_idx = torch.sym_min(
                        target_rank * full_chunk_size, partition_size
                    )
                    end_idx = torch.sym_min(start_idx + full_chunk_size, partition_size)
                    slice_size = end_idx - start_idx

                if slice_size > 0:
                    # Extract slice for target rank
                    if isinstance(partition_size, int):
                        rank_offset = min(my_rank * full_chunk_size, partition_size)
                        relative_start = max(0, start_idx - rank_offset)
                        slice_to_send = chunk.narrow(
                            self.dim, relative_start, slice_size
                        )
                    else:
                        # SymInt case - use narrow directly
                        rank_offset = torch.sym_min(
                            my_rank * full_chunk_size, partition_size
                        )
                        relative_start = torch.sym_max(0, start_idx - rank_offset)
                        slice_to_send = chunk.narrow(
                            self.dim, relative_start, slice_size
                        )

                    rank_chunks.append(slice_to_send)

            # Concatenate all slices for this target rank
            if rank_chunks:
                concatenated = torch.cat(rank_chunks, dim=self.dim)
            else:
                # Send empty tensor
                empty_shape = list(local_tensor.shape)
                empty_shape[self.dim] = 0
                concatenated = local_tensor.new_empty(empty_shape)

            scatter_list.append(concatenated)

        # Step 5: Perform all-to-all
        gathered_chunks = []
        torch.distributed.all_to_all(
            gathered_chunks, scatter_list, group=mesh.get_group(mesh_dim)
        )

        # Step 6: Concatenate received chunks in unaligned order
        valid_chunks = [chunk for chunk in gathered_chunks if chunk.size(self.dim) > 0]

        if not valid_chunks:
            # All chunks are empty
            result_shape = list(local_tensor.shape)
            result_shape[self.dim] = 0
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        return torch.cat(valid_chunks, dim=self.dim)

    # Utility methods and properties
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartitionedShard):
            return False
        return (
            self.dim == other.dim
            and self.num_partitions == other.num_partitions
            and self.splits == other.splits
            and self.aligned == other.aligned
        )

    def __hash__(self) -> int:
        return hash((self.dim, self.num_partitions, tuple(self.splits), self.aligned))

    def __repr__(self) -> str:
        """Machine readable representation of the PartitionedShard placement"""
        return (
            f"PartitionedShard(dim={self.dim}, num_partitions={self.num_partitions}, "
            f"splits={self.splits}, aligned={self.aligned})"
        )

    def __str__(self) -> str:
        """Human readable representation of the PartitionedShard placement"""
        alignment_str = "A" if self.aligned else "U"
        return f"PS({self.dim}, {self.num_partitions}, {alignment_str})"

    def is_shard(self, dim: Optional[int] = None) -> bool:
        """Check if this is a shard placement type (compatibility with base Placement)"""
        if dim is not None:
            return self.dim == dim
        return True

    def is_partitioned_shard(
        self, dim: Optional[int] = None, aligned: Optional[bool] = None
    ) -> bool:
        """Check if this is a partitioned shard with optional dimension and alignment checks"""
        if dim is not None and self.dim != dim:
            return False
        if aligned is not None and self.aligned != aligned:
            return False
        return True

    def get_total_size(self) -> Union[int, torch.SymInt]:
        """Get the total size across all partitions"""
        return sum(self.splits)

    def with_alignment(self, aligned: bool) -> "PartitionedShard":
        """Create a new PartitionedShard with different alignment"""
        return PartitionedShard(
            dim=self.dim,
            num_partitions=self.num_partitions,
            splits=self.splits,
            aligned=aligned,
        )
