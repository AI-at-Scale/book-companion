import argparse

def calculate_inference_tco(qps: float, latency_ms: float, gpu_monthly_cost: float, num_gpus: int):
    """
    Calculates KV Cache sizes and Token Economics.
    """
    print("=" * 50)
    print(f"🚀 AI at Scale: Inference Cost & KV Cache Planner")
    print("=" * 50)

    # Context calculations
    num_layers = 80
    num_heads = 64
    head_dim = 128
    precision_bytes = 2 # FP16/BF16

    bytes_per_token = 2 * num_layers * num_heads * head_dim * precision_bytes
    print(f"Model Assumed: Llama-3-70B architecture")
    print(f"KV Cache per Token: {bytes_per_token / 1024 / 1024:.3f} MB")

    test_seq_len = 8192
    cache_per_req_mb = (bytes_per_token * test_seq_len) / (1024 * 1024)
    print(f"KV Cache for {test_seq_len} tokens: {cache_per_req_mb:.1f} MB per request")
    
    print("-" * 50)
    print("💸 Economics (TCO per 1M Tokens)")
    
    # 1 GPU generates roughly 2000 tokens/sec total throughput continuously
    tokens_per_month = (qps * 1000) * 3600 * 24 * 30 # roughly 
    total_cost = num_gpus * gpu_monthly_cost
    cost_per_1m = (total_cost / tokens_per_month) * 1_000_000 if tokens_per_month > 0 else 0
    
    print(f"Assumed Throughput: {qps * 1000:,.0f} tokens/sec across {num_gpus} GPUs")
    print(f"Total Monthly Hardware Cost: ${total_cost:,.2f}")
    if cost_per_1m > 0:
        print(f"Your True Cost per 1M Tokens: ${cost_per_1m:.3f}")
    else:
        print("Throughput too low to calculate cost efficiently.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=float, default=10.0, help="Queries per second (assumes 1k output tokens/req)")
    parser.add_argument("--latency", type=float, default=50.0, help="Target latency per generation step in ms")
    parser.add_argument("--gpu_cost", type=float, default=1500.0, help="Monthly amortized cost of 1 GPU")
    parser.add_argument("--gpus", type=int, default=8, help="Number of inference GPUs")
    args = parser.parse_args()
    
    calculate_inference_tco(args.qps, args.latency, args.gpu_cost, args.gpus)
