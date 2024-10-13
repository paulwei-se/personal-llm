import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        embeddings = self.model.encode(documents)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(np.array(embeddings).astype('float32'))

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'document': self.documents[idx],
                'score': 1 - distances[0][i] / 2  # Convert L2 distance to similarity score
            })
        
        return results

# Usage example
if __name__ == "__main__":
    searcher = SemanticSearch()
    
    # Add some sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "I think, therefore I am"
    ]
    searcher.add_documents(documents)
    
    # Perform a search
    results = searcher.search("animal movement")
    
    print("Search Results:")
    for result in results:
        print(f"Document: {result['document']}")
        print(f"Score: {result['score']:.4f}")
        print()
