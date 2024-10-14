import asyncio
from transformers import pipeline

class QASummarizer:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")
        self.summarizer_pipeline = pipeline("summarization")
        self.document_models = {}

    async def update_model(self, doc_id, model_path):
        self.document_models[doc_id] = pipeline("question-answering", model=model_path, tokenizer=model_path)

    async def answer_question(self, context, question, doc_id=None):
        if doc_id in self.document_models:
            qa_pipeline = self.document_models[doc_id]
        else:
            qa_pipeline = self.qa_pipeline

        result = await asyncio.to_thread(qa_pipeline, question=question, context=context)
        return {
            'answer': result['answer'],
            'score': result['score']
        }

    async def summarize_text(self, text, max_length=150, min_length=50):
        summary = await asyncio.to_thread(self.summarizer_pipeline, text, max_length=max_length, min_length=min_length, do_sample=False)
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
