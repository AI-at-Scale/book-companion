import dspy
from dspy.teleprompt import BootstrapFewShot

# 1. Define the Signature (Input/Output Contract)
class RAGSignature(dspy.Signature):
    """Retrieve context and answer the question."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# 2. Define the Module (The Logic)
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

if __name__ == "__main__":
    print("This is a template compiling an Agent with DSPy.")
    print("To run properly, configure the LM and Retrieval model first:")
    print("lm = dspy.LM('openai/gpt-4o-mini')")
    print("rm = dspy.RM('you/retriever')")
    print("dspy.configure(lm=lm, rm=rm)")
    print("\n# Usage:")
    print('# optimizer = BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)')
    print('# compiled_rag = optimizer.compile(RAGModule(), trainset=train_data)')
    print('# result = compiled_rag("What is the capital of France?")')
