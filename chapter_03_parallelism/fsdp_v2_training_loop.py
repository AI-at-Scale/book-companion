import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.device_mesh import init_device_mesh

# Example placeholders
DP_DEGREE = 2
TP_DEGREE = 4

def fsdp_init(model):
    """
    Initializes a PyTorch model with Hybrid FSDP (Shards within node, Replicates across nodes)
    Requires a distributed environment (e.g., torchrun).
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Establish device mesh for 3D parallelism
    # Dimension 0: Data Parallel (Size: DP_DEGREE)
    # Dimension 1: Tensor Parallel (Size: TP_DEGREE) 
    device_mesh = init_device_mesh("cuda", (DP_DEGREE, TP_DEGREE))

    # Mixed Precision Policy (BF16 for compute, FP32 for reduction)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16, 
        reduce_dtype=torch.float32, 
        buffer_dtype=torch.bfloat16
    )

    # Wrap model with Hybrid Sharding
    # This is critical for minimizing inter-node communication
    model = FSDP(
        model,
        device_mesh=device_mesh['dp'], # Shard mainly along DP axis
        sharding_strategy=ShardingStrategy.HYBRID_SHARD, 
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
        sync_module_states=True
    )
    return model

if __name__ == "__main__":
    print("Run this script using torchrun to test FSDP initialization.")
    print("Example: torchrun --nproc_per_node=8 fsdp_init.py")
