from collections import deque

class Request:
    def __init__(self, request_id, prompt_len, max_gen_len):
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.max_gen_len = max_gen_len
        self.generated_tokens = 0
        
    def is_finished(self):
        return self.generated_tokens >= self.max_gen_len
        
    def memory_usage(self):
        return self.prompt_len + self.generated_tokens
        
    def initial_memory(self):
        return self.prompt_len

class MockEngine:
    def decode(self, batch):
        # Simulate generating 1 token for all requests in batch
        for req in batch:
            req.generated_tokens += 1
            print(f"Request {req.request_id} generated token: {req.generated_tokens}/{req.max_gen_len}")

class ContinuousScheduler:
    """
    A simplified Continuous Batching Scheduler algorithm.
    Frameworks like vLLM use a similar token budget to manage in-flight requests.
    """
    def __init__(self, max_tokens):
        self.running_batch = []
        self.waiting_queue = deque()
        self.token_budget = max_tokens
        self.engine = MockEngine()

    def has_budget(self, req):
        # Check if we have RAM for the request's initial tokens
        return self.token_budget >= req.initial_memory()

    def step(self):
        # 1. Eject finished sequences
        finished = [s for s in self.running_batch if s.is_finished()]
        for s in finished:
            print(f"✅ Request {s.request_id} finished. Reclaiming {s.memory_usage()} tokens.")
            self.running_batch.remove(s)
            self.token_budget += s.memory_usage()

        # 2. Schedule new requests if budget permits
        while self.waiting_queue:
            next_req = self.waiting_queue[0]
            if self.has_budget(next_req):
                self.waiting_queue.popleft()
                self.running_batch.append(next_req)
                self.token_budget -= next_req.initial_memory()
                print(f"🚀 Started Request {next_req.request_id}")
            else:
                break # Memory full, wait for next step

        # 3. Decode one step for all running sequences
        if self.running_batch:
            self.engine.decode(self.running_batch)
            # Subtract 1 token budget for each generated token across the batch
            self.token_budget -= len(self.running_batch)

if __name__ == "__main__":
    scheduler = ContinuousScheduler(max_tokens=100)
    
    # Queue up 3 requests of varying lengths
    scheduler.waiting_queue.append(Request(1, prompt_len=10, max_gen_len=5))
    scheduler.waiting_queue.append(Request(2, prompt_len=20, max_gen_len=3))
    scheduler.waiting_queue.append(Request(3, prompt_len=15, max_gen_len=6))
    
    print("Starting continuous batching loop...")
    for i in range(10):
        print(f"\n--- Step {i+1} ---")
        scheduler.step()
