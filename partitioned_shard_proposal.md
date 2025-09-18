# Proposal for PartitionedShard Placement Type

## Overview

This proposal introduces a new placement type `PartitionedShard(Placement)` to the PyTorch distributed tensor framework. This placement type extends the existing sharding capabilities to handle partitioned shards with variable chunk sizes and supports both aligned and unaligned partitioning strategies.

## Motivation

The current `Shard` placement type assumes uniform chunk sizes across all ranks in a mesh dimension. However, for certain use cases such as Mixture of Experts (MoE) models, we need the ability to handle variable-sized partitions within shards. The `PartitionedShard` placement enables efficient distribution of tensors where different partitions may have different sizes while maintaining the semantic structure needed for MoE operations.

## Class Definition

```python
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
```

## Semantic Description

### Core Concepts

1. **Partitions**: Logical subdivisions of the tensor along the specified dimension
2. **Shards**: Physical distribution units across mesh dimensions
3. **Splits**: Size specification for each partition (not indices, but element counts)
4. **Alignment**: Strategy for how partitions are distributed across shards

### Chunk Size Calculation
- The chunk sizes are similar to standard `Shard` where the number of chunks equals the mesh dimension size
- Each chunk can have even or uneven sizes depending on partition alignment
- Total elements across all partitions must equal the tensor dimension size

### Alignment Strategies

#### Unaligned Partitioned Shard
- **Definition**: Each shard contains slices of ALL partitions
- **Example**: With 2 shards and 4 partitions (P00, P01, P02, P03 for shard 0; P10, P11, P12, P13 for shard 1):
  - Shard 1: [P00, P01, P02, P03]
  - Shard 2: [P10, P11, P12, P13]
- **Use Case**: When partitions need to be processed together within each shard

#### Aligned Partitioned Shard
- **Definition**: Each shard contains complete partitions (one or more full partitions)
- **Partition Distribution**: `num_partitions_per_shard = num_partitions // num_shards`
- **Example**: With same setup:
  - Shard 1: [P00, P10, P01, P11]
  - Shard 2: [P02, P12, P03, P13]
- **Use Case**: When partitions can be independently processed across shards

## Core Operations

### 1. Unaligned to Replicate Conversion

```python
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
    1. All-gather to collect all shards: [P00, P01, P02, P03, P10, P11, P12, P13]
    2. Perform partition alignment using splits to get: [P00, P10, P01, P11, P02, P12, P03, P13]
    3. Return fully reconstructed tensor
    """
```

### 2. Aligned to Replicate Conversion

```python
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
    1. All-gather with list of tensors (handles dynamic sizes): [P00, P10, P01, P11, P02, P12, P03, P13]
    2. Concatenate to form complete tensor
    """
```

### 3. Replicate to Shard Conversion

```python
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
```

### 4. Alignment Conversion Operations

```python
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
       - Input splits (shard1): [4,6,4,2], (shard2): [2,4,8,2]
       - Output splits (shard1): [4,6,2,4], (shard2): [4,2,8,2]
    3. Compute boundaries:
       - in_boundaries = input_splits.reshape(num_shards, num_partitions_per_shard).sum(dim=1)
       - out_boundaries = out_splits.reshape(num_shards, num_partitions_per_shard).sum(dim=1)
    4. Second all-to-all: Exchange tensor data using boundaries
    5. Local reordering using out_splits to achieve final alignment
    """

def _aligned_to_unaligned_shard(
    self,
    local_tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> torch.Tensor:
    """
    Convert aligned partitioned shard to unaligned partitioned shard.

    This performs the reverse operation of unaligned_to_aligned_shard.

    Algorithm:
    Starting state (aligned):
    - Shard 1: [P00, P10, P01, P11] with splits [4,2,6,4]
    - Shard 2: [P02, P12, P03, P13] with splits [4,8,2,2]

    Goal (unaligned):
    - Shard 1: [P00, P01, P02, P03] with splits [4,6,4,2]
    - Shard 2: [P10, P11, P12, P13] with splits [2,4,8,2]

    Steps:
    1. Calculate partitions per shard: num_partitions_per_shard = num_partitions / mesh_size
    2. Prepare current split information (what we currently have per shard)
       - current_splits = local tensor partition sizes in aligned order
       - Example: Shard1 current_splits = [4,2,6,4], Shard2 current_splits = [4,8,2,2]
    3. First all-to-all: Exchange split information to get target unaligned splits
       - We need to transpose the split matrix from aligned to unaligned layout
       - Input: splits arranged as [shard][partition_within_shard]
       - Output: splits arranged as [partition][shard]
       - After exchange: Shard1 gets [4,6,4,2], Shard2 gets [2,4,8,2]
    4. Compute boundaries for data exchange:
       - in_boundaries: Current chunk boundaries (aligned layout)
       - out_boundaries: Target chunk boundaries (unaligned layout)
       - in_boundaries = current_splits.reshape(num_partitions_per_shard, num_shards).sum(dim=0)
       - out_boundaries = target_splits  # Sequential partitions per shard
    5. Second all-to-all: Exchange tensor data using computed boundaries
       - Send data from aligned layout to unaligned layout
       - Each rank sends: partitions intended for other ranks' unaligned layout
       - Each rank receives: sequential partitions for unaligned layout
    6. Local reordering if needed to ensure correct partition order

    Detailed Example:
    Input (aligned): Shard1=[P00:4, P10:2, P01:6, P11:4], Shard2=[P02:4, P12:8, P03:2, P13:2]
    After step 3: target_splits Shard1=[4,6,4,2], Shard2=[2,4,8,2]
    After step 5: Shard1=[P00:4, P01:6, P02:4, P03:2], Shard2=[P10:2, P11:4, P12:8, P13:2]
    Final (unaligned): Achieved target layout
    """
```
