from typing import List, Optional, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import fill_empty_tensor_to_shards
from torch.distributed.tensor.placement_types import Shard


class DynamicShard(Shard):
    def __init__(self, dim: int, chunk_sizes: List[Union[int, torch.SymInt]]):
        """
        DynamicShard allows variable-length chunks along the specified tensor dimension.

        Args:
            dim (int): The tensor dimension to shard.
            chunk_sizes (List[Union[int, torch.SymInt]]):
                List of chunk sizes for each device in the mesh dimension.
                The length of chunk_sizes must equal the mesh dimension size.
                A chunk size of 0 creates an empty tensor placeholder for that rank.
                Chunk sizes can be int or torch.SymInt to support symbolic shapes.
        """
        super().__init__(dim)
        if not isinstance(chunk_sizes, (list, tuple)):
            raise TypeError(
                "chunk_sizes must be a list or tuple of ints or torch.SymInt"
            )
        if len(chunk_sizes) == 0:
            raise ValueError("chunk_sizes must not be empty")

        # Validate chunk size types
        for i, size in enumerate(chunk_sizes):
            if not (isinstance(size, int) or isinstance(size, torch.SymInt)):
                raise TypeError(
                    f"Chunk size at index {i} must be an int or torch.SymInt, got {type(size)}"
                )

        # Validate non-negative chunk sizes (for concrete values)
        for i, size in enumerate(chunk_sizes):
            if isinstance(size, int) and size < 0:
                raise ValueError(
                    f"Chunk size at index {i} must be non-negative, got {size}"
                )

        self.chunk_sizes = list(chunk_sizes)

    def _compute_offsets(self) -> List[Union[int, torch.SymInt]]:
        """
        Compute offsets from chunk sizes using cumulative sum.
        Returns offsets including the final endpoint.
        """
        if not self.chunk_sizes:
            return [0]

        offsets: List[Union[int, torch.SymInt]] = [0]
        current_offset = 0
        for size in self.chunk_sizes:
            current_offset += size
            offsets.append(current_offset)
        return offsets

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        chunk_sizes: List[Union[int, torch.SymInt]],
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Split tensor into variable-sized chunks based on provided chunk_sizes.
        Updates the placement's chunk_sizes to the new values.

        Note: with_padding is ignored as padding defeats the purpose of DynamicShard.
              All chunks maintain their original variable sizes.

        Args:
            tensor: Tensor to split
            chunk_sizes: New list of chunk sizes for splitting the tensor
            with_padding: Ignored - DynamicShard never pads
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

        # Update the placement's chunk_sizes to the new values
        self.chunk_sizes = list(chunk_sizes)

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

        # Fill empty tensors if needed (compatibility with base class)
        tensor_list = fill_empty_tensor_to_shards(
            tensor_list, self.dim, len(chunk_sizes) - len(tensor_list)
        )

        # DynamicShard never pads - return zero pad sizes
        pad_sizes = [0] * len(tensor_list)
        return tensor_list, pad_sizes

    def chunk_slices(self, tensor_shape):
        """
        Returns a list of slice objects for each chunk along the sharded dimension,
        based on the chunk sizes.

        Args:
            tensor_shape (Sequence[int]): The shape of the tensor to be sharded.

        Returns:
            List[tuple]: List of slice tuples for each chunk.
        """
        if self.dim >= len(tensor_shape):
            raise ValueError(
                f"Sharding dim {self.dim} >= tensor ndim {len(tensor_shape)}"
            )

        # Compute offsets from chunk sizes
        offsets = self._compute_offsets()

        slices = []
        for i in range(len(self.chunk_sizes)):
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicShard):
            # For compatibility, check if it's a regular Shard with same dim
            if isinstance(other, Shard):
                return self.dim == other.dim
            return False
        return self.dim == other.dim and self.chunk_sizes == other.chunk_sizes

    def __hash__(self) -> int:
        # Use tuple of chunk_sizes for hashing, but handle SymInt carefully
        try:
            chunk_sizes_tuple = tuple(self.chunk_sizes)
            return hash((self.dim, chunk_sizes_tuple))
        except TypeError:
            # If chunk_sizes contain unhashable SymInt, fall back to dim only
            return hash(self.dim)

    def __repr__(self) -> str:
        """Machine readable representation of the DynamicShard placement"""
        return f"DynamicShard(dim={self.dim}, chunk_sizes={self.chunk_sizes})"

    def __str__(self) -> str:
        """Human readable representation of the DynamicShard placement"""
        return f"DS({self.dim}, {len(self.chunk_sizes)})"

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Shard and scatter a tensor on a mesh dimension with variable chunk sizes.
        Uses torch.distributed.scatter which can handle variable-sized tensors.
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # if rank is not part of mesh, return empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # Skip communications, just split locally
            scatter_list, _ = self._split_tensor(
                tensor, self.chunk_sizes, with_padding=False, contiguous=True
            )
            return scatter_list[mesh_dim_local_rank]

        # Get the process group for this mesh dimension
        group = mesh.get_group(mesh_dim)

        if mesh_dim_local_rank == src_data_rank:
            # Source rank: split tensor and create scatter_list
            scatter_list, _ = self._split_tensor(
                tensor, self.chunk_sizes, with_padding=False, contiguous=True
            )
        else:
            # Non-source ranks: scatter_list should be None
            scatter_list = None

        # Create output tensor for this rank's chunk
        chunk_size = self.chunk_sizes[mesh_dim_local_rank]
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
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Reduce and scatter with variable chunk sizes.
        Cannot use reduce_scatter_tensor since chunks are not equal-sized.
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
        chunk_size = self.chunk_sizes[mesh_dim_local_rank]

        if chunk_size == 0:
            # Return empty tensor
            chunk_shape = list(tensor.shape)
            chunk_shape[self.dim] = 0
            return tensor.new_empty(chunk_shape, requires_grad=tensor.requires_grad)

        # Compute offset for this rank
        offset = sum(self.chunk_sizes[:mesh_dim_local_rank])
        result = reduced_tensor.narrow(self.dim, offset, chunk_size)

        return result.contiguous()

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        All-gather variable-sized shards to reconstruct the full replicated tensor.
        Uses torch.distributed.all_gather which supports uneven sized tensors.
        """
        my_coordinate = mesh.get_coordinate()

        if my_coordinate is None:
            # Create empty tensor with the logical shape
            result_shape = list(current_logical_shape)
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
        # TODO: Need to define a functional version of all_gather to use here
        torch.distributed.all_gather(gathered_tensors, local_tensor, group=group)

        # Filter out empty chunks and concatenate
        valid_chunks = [chunk for chunk in gathered_tensors if chunk.size(self.dim) > 0]

        if not valid_chunks:
            # All chunks are empty
            result_shape = list(current_logical_shape)
            return local_tensor.new_empty(
                result_shape, requires_grad=local_tensor.requires_grad
            )

        result = torch.cat(valid_chunks, dim=self.dim)
        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        Transform from replicated tensor to a sharded tensor by taking the appropriate chunk.
        """
        shards, _ = self._split_tensor(
            local_tensor,
            self.chunk_sizes,
            with_padding=False,
            contiguous=False,
        )
        return shards[shard_index].clone()

    @staticmethod
    def _local_shard_size_and_offset(
        curr_local_size: int,
        num_chunks: int,
        rank: int,
    ) -> tuple[int, int]:
        """
        Not supported for DynamicShard as it uses variable chunk sizes.
        Use chunk_sizes directly instead of computing from equal-sized assumptions.
        """
        raise NotImplementedError(
            "DynamicShard does not support _local_shard_size_and_offset as it uses "
            "variable chunk sizes. Use the chunk_sizes attribute directly instead."
        )

    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        Not supported for DynamicShard as resharding with variable chunk sizes
        requires custom logic that depends on the specific use case.
        """
        raise NotImplementedError(
            "DynamicShard does not support _to_new_shard_dim as resharding with "
            "variable chunk sizes requires domain-specific logic. Consider "
            "redistributing to Replicate first, then applying new DynamicShard."
        )
