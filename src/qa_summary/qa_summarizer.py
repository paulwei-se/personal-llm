from transformers import pipeline

class QASummarizer:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")
        self.summarizer_pipeline = pipeline("summarization")

    def answer_question(self, context, question):
        """Answer a question based on the given context."""
        result = self.qa_pipeline(question=question, context=context)
        return {
            'answer': result['answer'],
            'score': result['score']
        }

    def summarize_text(self, text, max_length=150, min_length=50):
        """Generate a summary of the given text."""
        summary = self.summarizer_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

# Usage example
if __name__ == "__main__":
    qa_sum = QASummarizer()
    
    context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world."
    
    question = "Who designed the Eiffel Tower?"
    answer = qa_sum.answer_question(context, question)
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']} (Score: {answer['score']:.2f})")
    
    summary = qa_sum.summarize_text(context)
    print(f"\nSummary: {summary}")
