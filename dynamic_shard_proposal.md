# Proposal for DynamicShard Placement Type

## Overview

This proposal introduces a new placement type `DynamicShard(Placement)` to the PyTorch distributed tensor framework. This placement type extends the existing sharding capabilities to handle variable-sized chunks along a tensor dimension, enabling efficient distribution of tensors with non-uniform sharding requirements.

## Motivation

The current `Shard` placement type assumes equal-sized chunks across all ranks in a mesh dimension, following `torch.chunk()` semantics where the last few shards might be smaller when the tensor dimension is not evenly divisible. However, for certain use cases (eg. token-choice routing in MoEs), we may need to distribute tensors with non-uniform sharding requirements, where each rank receives a chunk of a different size based on its specific needs.


## Class Definition

```python
class DynamicShard(Placement):
    """
    DynamicShard allows variable-length chunks along the specified tensor dimension.

    Unlike the standard Shard placement which assumes equal-sized chunks,
    DynamicShard handles variable chunk sizes across a mesh dimension.

    Args:
        dim (int): The tensor dimension to shard.
    """

    def __init__(self, dim: int):
        self.dim = dim
```

## Semantic Description

### Core Concepts

1. **Variable Chunks**: Each rank in the mesh dimension receives a chunk of potentially different size
2. **Explicit Sizing**: Chunk sizes are explicitly provided as parameters to methods, not inferred from tensor dimensions
3. **No Padding**: Unlike standard `Shard`, `DynamicShard` never pads tensors to maintain exact variable sizes
4. **Exact Distribution**: The sum of all chunk sizes must exactly equal the tensor dimension size

### Key Differences from Standard Shard

| Feature | Standard Shard | DynamicShard |
|---------|----------------|--------------|
| **Chunk Sizes** | Equal (with possible smaller last chunk) | Variable, explicitly specified |
| **Padding** | Uses padding for collective operations | Never pads, maintains exact sizes |
| **Size Specification** | Inferred from tensor and mesh size | Explicitly provided as parameters |
| **Use Case** | General uniform distribution | Specialized non-uniform distribution |

## Core Operations

### 1. Tensor Splitting

```python
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

    Key Features:
    - Supports both int and torch.SymInt for dynamic shapes
    - Validates that sum of chunk_sizes equals tensor dimension size
    - Handles empty chunks (size 0) gracefully
    - Never applies padding (with_padding parameter ignored)
    - Returns (chunk_list, pad_sizes) where pad_sizes is always [0, 0, ...]
    """
```


## Key Algorithms

### _split_tensor Algorithm
1. **Validate**: Check chunk_sizes sum equals tensor dimension size
2. **Split**: Use `tensor.narrow(dim, offset, size)` for each chunk
3. **Handle Empty**: Create empty tensors for zero-sized chunks
4. **Return**: (chunk_list, [0, 0, ...]) - no padding applied

### _shard_tensor Algorithm
1. **Source Rank**: Split tensor using `_split_tensor()`
2. **All Ranks**: Allocate output tensor based on expected chunk size
3. **Distribute**: Use `torch.distributed.scatter()` for variable-size chunks

### 2. Chunk Slice Computation

```python
def _chunk_slices(
    self,
    tensor_shape: Sequence[int],
    chunk_sizes: List[Union[int, torch.SymInt]]
) -> List[tuple]:
    """
    Returns slice objects for each chunk along the sharded dimension.

    Features:
    - Computes exact slice boundaries using cumulative offsets
    - Validates slice boundaries are within tensor bounds
    - Supports both concrete and symbolic chunk sizes
    """
```

### 3. Distributed Sharding

```python
def _shard_tensor(
    self,
    tensor: torch.Tensor,
    chunk_sizes: List[Union[int, torch.SymInt]],
    mesh: DeviceMesh,
    mesh_dim: int,
    src_data_rank: Optional[int] = 0,
) -> torch.Tensor:
    """
    Shard and scatter tensor with variable chunk sizes.

    Communication Pattern:
    - Uses torch.distributed.scatter() which handles variable-sized tensors
    - Source rank splits tensor according to chunk_sizes
    - Each rank receives its designated chunk size
    """
```

### 4. Reduction with Variable Sizes

```python
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

    Implementation Strategy:
    Since reduce_scatter_tensor requires equal-sized chunks, DynamicShard
    implements this as:
    1. All-reduce the full tensor
    2. Each rank locally extracts their designated chunk
    """
```


### 5. Replication Operations

```python
def _to_replicate_tensor(
    self,
    local_tensor: torch.Tensor,
    chunk_sizes: List[Union[int, torch.SymInt]],
    mesh: DeviceMesh,
    mesh_dim: int,
) -> torch.Tensor:
    """
    All-gather variable-sized shards to reconstruct full tensor.

    Features:
    - Uses torch.distributed.all_gather() with variable-sized tensors
    - Handles empty chunks gracefully
    - Concatenates gathered chunks to form complete tensor
    """

def _replicate_to_shard(
    self,
    local_tensor: torch.Tensor,
    chunk_sizes: List[Union[int, torch.SymInt]],
    mesh: DeviceMesh,
    mesh_dim: int,
    shard_index: int,
) -> torch.Tensor:
    """
    Transform replicated tensor to variable-sized shard.

    Simple Implementation:
    - Split tensor according to chunk_sizes
    - Return the chunk corresponding to shard_index
    """
```
