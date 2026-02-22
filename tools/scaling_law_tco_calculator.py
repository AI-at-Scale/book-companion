import argparse
import numpy as np

def calculate_chinchilla_optimal(compute_budget_flops: float):
    """
    Calculate optimal N (parameters) and D (tokens) given compute budget C.
    Chinchilla Rule: D ≈ 20N (Hoffmann et al., 2022)
    C ≈ 6 * N * D ≈ 120 * N^2
    N = sqrt(C / 120)
    """
    n_optimal = np.sqrt(compute_budget_flops / 120)
    d_optimal = 20 * n_optimal
    return n_optimal, d_optimal

def estimate_tco(budget_usd: float, gpu_hourly_rate: float, flops_per_sec: float):
    """
    Estimate total FLOPs budget based on dollars and GPU rate.
    """
    gpu_hours = budget_usd / gpu_hourly_rate
    flops_per_hour = flops_per_sec * 3600
    total_flops = gpu_hours * flops_per_hour
    return total_flops, gpu_hours

def main():
    parser = argparse.ArgumentParser(description="AI at Scale - Scaling Law & TCO Calculator")
    parser.add_argument("--budget", type=float, default=100000, help="Total training budget in USD.")
    parser.add_argument("--gpu_rate", type=float, default=4.00, help="Hourly cost per GPU in USD (e.g., $4.00 for H100 spot).")
    parser.add_argument("--gpu_flops", type=float, default=1.0e15, help="Effective FLOPs/sec per GPU. Default is 1 PFLOPS (BF16 H100 realistic).")
    args = parser.parse_args()

    print("="*50)
    print(f"💰 Training Budget: ${args.budget:,.2f}")
    print(f"🖥️  GPU Rate: ${args.gpu_rate:.2f}/hr")
    print(f"⚡ Effective FLOPs/sec: {args.gpu_flops:.2e}")
    print("="*50)

    total_flops, gpu_hours = estimate_tco(args.budget, args.gpu_rate, args.gpu_flops)
    print(f"🕒 Total Available GPU-Hours: {gpu_hours:,.0f} hours")
    print(f"🧮 Total Compute Budget (C): {total_flops:.2e} FLOPs")
    
    n_opt, d_opt = calculate_chinchilla_optimal(total_flops)
    
    print("-" * 50)
    print("📈 Chinchilla Compute-Optimal Projection:")
    print(f"   Optimal Parameters (N): {n_opt/1e9:.2f} Billion")
    print(f"   Optimal Tokens (D):     {d_opt/1e9:.2f} Billion")
    print("="*50)
    print("Note: If you plan to serve this model heavily, post-Chinchilla scaling laws ")
    print("suggest training a smaller model (e.g., 50% of N) on far more data (e.g., 200% of D).")

if __name__ == "__main__":
    main()
