import random
import time

class MockModel:
    def __init__(self, name, throughput_tok_sec):
        self.name = name
        self.throughput = throughput_tok_sec

    def generate_token(self):
        # Simulate time taken to generate a single token (sequential)
        time.sleep(1 / self.throughput)
        return random.randint(1, 1000)

    def validate_tokens(self, tokens):
        # Simulate a parallel forward pass (validating multiple tokens at once)
        # In real Speculative Decoding, this takes roughly the same time as generating 1 token
        time.sleep(1 / self.throughput)
        
        # Randomly decide where the draft model "guessed wrong"
        # High probability of being right for a good draft model
        for i in range(len(tokens)):
            if random.random() > 0.85: # 85% accuracy guess
                return i, random.randint(1, 1000) # Rejection index and correction
        return len(tokens), None

def speculative_decoding(draft_model, target_model, K=4):
    """
    Standard Speculative Decoding Algorithm.
    The Draft model generates K tokens sequentially.
    The Target model validates them in a single parallel pass.
    """
    generated_tokens = []
    total_steps = 0
    start_time = time.time()

    print(f"--- Starting Speculative Decoding (K={K}) ---")
    print(f"Draft: {draft_model.name}, Target: {target_model.name}\n")

    while len(generated_tokens) < 20:
        total_steps += 1
        
        # 1. Draft model generates K tokens sequentially
        draft_guesses = []
        for _ in range(K):
            draft_guesses.append(draft_model.generate_token())

        # 2. Target model validates guesses in one parallel forward pass
        n_accepted, correction = target_model.validate_tokens(draft_guesses)
        
        # 3. Acceptance Logic
        accepted_this_step = draft_guesses[:n_accepted]
        generated_tokens.extend(accepted_this_step)
        
        if correction is not None:
            generated_tokens.append(correction)
            print(f"Step {total_steps}: Accepted {n_accepted} tokens. Rejected at index {n_accepted}. Applied correction.")
        else:
            print(f"Step {total_steps}: Accepted all {K} tokens!")

    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n✅ Finished! Generated {len(generated_tokens)} tokens in {total_duration:.2f} seconds.")
    print(f"Tokens per second: {len(generated_tokens)/total_duration:.2f}")

if __name__ == "__main__":
    # Small draft model (fast but less accurate)
    draft = MockModel("Llama-3-8B", throughput_tok_sec=150)
    
    # Large target model (slow sequential, but fast parallel validation)
    target = MockModel("Llama-3-405B", throughput_tok_sec=10)
    
    speculative_decoding(draft, target, K=4)
