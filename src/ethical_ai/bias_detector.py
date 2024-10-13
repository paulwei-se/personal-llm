from transformers import pipeline
import numpy as np

class BiasDetector:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")

    def detect_sentiment_bias(self, texts):
        """Detect potential sentiment bias in a list of texts."""
        sentiments = self.sentiment_analyzer(texts)
        positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        
        bias_score = abs(positive_count - negative_count) / len(texts)
        return {
            'bias_score': bias_score,
            'positive_ratio': positive_count / len(texts),
            'negative_ratio': negative_count / len(texts)
        }

    def detect_toxicity(self, texts):
        """Detect potential toxic content in a list of texts."""
        toxicity_scores = self.toxic_classifier(texts)
        toxic_count = sum(1 for t in toxicity_scores if t['label'] == 'toxic')
        
        return {
            'toxicity_ratio': toxic_count / len(texts),
            'toxic_count': toxic_count,
            'total_count': len(texts)
        }

# Usage example
if __name__ == "__main__":
    detector = BiasDetector()
    
    texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "The service was okay, nothing special.",
        "You're an idiot for buying this."
    ]
    
    sentiment_bias = detector.detect_sentiment_bias(texts)
    print("Sentiment Bias:", sentiment_bias)
    
    toxicity = detector.detect_toxicity(texts)
    print("Toxicity:", toxicity)
