from transformers import pipeline
import numpy as np

class AIExplainer:
    def __init__(self):
        self.qa_pipeline = pipeline("question-answering")

    def explain_answer(self, context, question, answer):
        """Provide a simple explanation for the AI-generated answer."""
        # Use the QA pipeline to get the answer and its score
        result = self.qa_pipeline(question=question, context=context)
        
        # Calculate the relevance of each word in the context to the answer
        words = context.split()
        word_scores = []
        for i, word in enumerate(words):
            temp_context = ' '.join(words[:i] + ['[MASK]'] + words[i+1:])
            temp_result = self.qa_pipeline(question=question, context=temp_context)
            word_scores.append(abs(result['score'] - temp_result['score']))
        
        # Normalize word scores
        word_scores = np.array(word_scores) / np.max(word_scores)
        
        # Get the top 5 most relevant words
        top_words = [words[i] for i in np.argsort(word_scores)[-5:][::-1]]
        
        explanation = f"The AI system found this answer with a confidence of {result['score']:.2f}. "
        explanation += f"The answer was mainly derived from these key words: {', '.join(top_words)}. "
        explanation += "Please note that this explanation is a simplification and the actual process is more complex."
        
        return explanation

# Usage example
if __name__ == "__main__":
    explainer = AIExplainer()
    
    context = "The Eiffel Tower, located in Paris, France, was completed in 1889. It was designed by engineer Gustave Eiffel and stands 324 meters tall."
    question = "When was the Eiffel Tower built?"
    answer = "The Eiffel Tower was completed in 1889."
    
    explanation = explainer.explain_answer(context, question, answer)
    print("Question:", question)
    print("Answer:", answer)
    print("Explanation:", explanation)
