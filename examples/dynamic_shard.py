from typing import List, Optional, Sequence, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._collective_utils import fill_empty_tensor_to_shards


class DynamicShard:
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
