import argparse
import math

def configure_3d_parallelism(model_params_billions: float, num_gpus: int, gpu_memory_gb: int):
    """
    Suggests a 3D parallelism strategy (DP, TP, PP) for a given model size and cluster.
    """
    print("=" * 50)
    print(f"📐 AI at Scale: 3D Parallelism Planner")
    print(f"🤖 Model Size: {model_params_billions:.1f} Billion Parameters")
    print(f"🖥️  Cluster: {num_gpus} GPUs ({gpu_memory_gb}GB VRAM each)")
    print("=" * 50)

    # 1. Calculate static memory requirements (Weights, Gradients, Optimizer states)
    # Typically 16 bytes per parameter (FP16 weights + FP32 optimizer + FP16 gradients) for ZeRO-1
    bytes_per_param = 16 
    total_memory_required_gb = (model_params_billions * 1e9 * bytes_per_param) / 1e9
    
    print(f"Total Theoretical VRAM Required: {total_memory_required_gb:.1f} GB")
    
    # 2. Heuristics for Topology
    tp_degree = 1
    pp_degree = 1
    dp_degree = 1

    # Tensor Parallelism (keep within a node, usually max 8 GPUs)
    if model_params_billions > 10:
        tp_degree = min(8, num_gpus)
        
    # Pipeline Parallelism (scale across nodes)
    memory_per_tp_group = tp_degree * gpu_memory_gb
    if total_memory_required_gb > memory_per_tp_group:
        # We need PP to fit the model
        pp_degree = math.ceil(total_memory_required_gb / memory_per_tp_group)
        # Round up to nearest power of 2 for simplicity
        pp_degree = 1 << (pp_degree - 1).bit_length()
        if pp_degree * tp_degree > num_gpus:
            print("❌ ERROR: Cluster does not have enough GPUs to fit this model.")
            return

    # Data Parallelism (the rest of the GPUs)
    dp_degree = num_gpus // (tp_degree * pp_degree)

    print("\n✅ Recommended 3D Sharding Topology:")
    print(f"   [TP] Tensor Parallel Degree:   {tp_degree} (Intra-node NVLink)")
    print(f"   [PP] Pipeline Parallel Degree: {pp_degree} (Inter-node RoCE/InfiniBand)")
    print(f"   [DP] Data Parallel Degree:     {dp_degree} (FSDP / ZeRO-1)")
    print("-" * 50)
    print("Note: If using ZeRO-3/FSDP full sharding, PP can often be set to 1,")
    print("but communication overhead will drastically increase.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=float, default=70.0, help="Model parameters in Billions (e.g., 70 for Llama-3-70B)")
    parser.add_argument("--gpus", type=int, default=64, help="Total number of GPUs available")
    parser.add_argument("--vram", type=int, default=80, help="VRAM per GPU in GB (e.g., 80 for H100)")
    args = parser.parse_args()
    
    configure_3d_parallelism(args.params, args.gpus, args.vram)
