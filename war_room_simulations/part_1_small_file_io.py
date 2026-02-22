import time
import random

def simulate_small_file_io(num_files: int):
    """
    Simulates the overhead of opening many small files during data preprocessing.
    """
    print(f"Simulating reading {num_files} small files from disk latency...")
    start = time.time()
    
    for _ in range(num_files):
        # Time to open file descriptor is high
        time.sleep(random.uniform(0.001, 0.003)) 
        
    end = time.time()
    print(f"Total time for small files: {end - start:.2f} seconds")
    
if __name__ == "__main__":
    simulate_small_file_io(1000)
