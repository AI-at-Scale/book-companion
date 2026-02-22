import time

def simulate_infinite_agent_loop(max_iterations: int = 10):
    """
    Simulates a cognitive agent getting stuck in an infinite planning loop.
    """
    print("Agent triggered. Attempting to solve task...")
    
    for i in range(max_iterations):
        print(f"Iteration {i}: Agent generated plan. Executing...")
        time.sleep(1)
        print("Execution failed. Agent adjusting plan...")
    
    print("CRITICAL TRIGGER: Agent forcefully halted by oversight node due to infinite loop detection.")

if __name__ == "__main__":
    simulate_infinite_agent_loop(5)
