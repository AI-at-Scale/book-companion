"""
Thermal Throttling Simulator: Modeling the Straggler Effect
Part of the "AI at Scale" Companion Code Repository.

This simulator demonstrates how localized thermal hot spots propagate through 
a synchronous distributed training cluster, causing a massive drop in 
effective TFLOPS due to the slowest node (straggler) limiting the entire group.
"""

import numpy as np
import time

def simulate_training_run(
    num_nodes=32, 
    gpus_per_node=8, 
    target_clock_mhz=1800, 
    throttle_clock_mhz=1300, 
    thermal_event_prob=0.05,
    iterations=100
):
    """
    Simulates the effective throughput of a synchronous training run.
    """
    total_gpus = num_nodes * gpus_per_node
    print(f"--- Simulating Training on Cluster: {num_nodes} Nodes ({total_gpus} GPUs) ---")
    print(f"Target Clock: {target_clock_mhz} MHz | Throttle Clock: {throttle_clock_mhz} MHz")
    print(f"Probability of a node overheating: {thermal_event_prob*100}%\n")

    # Base performance (all at target)
    theoretical_peak = total_gpus * target_clock_mhz
    
    performance_log = []

    for i in range(iterations):
        # Determine which nodes are throttled this step
        # Note: In a synchronous run (All-Reduce), the SPEED of the cluster 
        # is the speed of the SLOWEST GPU.
        
        node_clocks = []
        for n in range(num_nodes):
            if np.random.random() < thermal_event_prob:
                node_clocks.append(throttle_clock_mhz)
            else:
                node_clocks.append(target_clock_mhz)
        
        # The synchronous cluster speed is the minimum clock among all nodes
        cluster_speed_mhz = min(node_clocks)
        
        # Effective throughput
        effective_tflops_percent = (cluster_speed_mhz / target_clock_mhz) * 100
        performance_log.append(effective_tflops_percent)

    avg_perf = np.mean(performance_log)
    min_perf = np.min(performance_log)
    
    print(f"Results over {iterations} iterations:")
    print(f"Average Cluster Efficiency: {avg_perf:.2f}%")
    print(f"Worst-Case Efficiency: {min_perf:.2f}%")
    print(f"Total FLOPs Wasted: {100 - avg_perf:.2f}%\n")
    
    if avg_perf < 80:
        print("WARNING: High Thermal Straggler Effect detected!")
        print("Recommendation: Transition to Liquid Cooling or Heat-Aware Scheduling.")
    else:
        print("Cluster performance within acceptable thermal bounds.")

if __name__ == "__main__":
    # Simulate a small-to-mid scale enterprise run
    simulate_training_run(num_nodes=64, thermal_event_prob=0.02)
    
    print("\n--- Extreme Scaling Scenario (2000 Nodes) ---")
    # In a 2000-node cluster, even a 0.1% chance of a node overheating 
    # almost guarantees a straggler every single step.
    simulate_training_run(num_nodes=2000, thermal_event_prob=0.001)
