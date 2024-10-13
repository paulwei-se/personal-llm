import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticSearch:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.index = None
        self.documents = []
        self.document_languages = []

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            self.logger.warning(f"Could not detect language. Defaulting to English.")
            return 'en'

    def add_documents(self, documents):
        embeddings = []
        for doc in documents:
            lang = self.detect_language(doc)
            self.documents.append(doc)
            self.document_languages.append(lang)
            embedding = self.model.encode([doc])[0]
            embeddings.append(embedding)

        embeddings = np.array(embeddings).astype('float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(embeddings)
        self.logger.info(f"Added {len(documents)} documents to the index")

    def search(self, query, k=5):
        query_lang = self.detect_language(query)
        query_embedding = self.model.encode([query])

        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure the index is valid
                results.append({
                    'document': self.documents[idx],
                    'score': 1 - distances[0][i] / 2,  # Convert L2 distance to similarity score
                    'language': self.document_languages[idx]
                })
        
        self.logger.info(f"Performed search with query in {query_lang}. Found {len(results)} results.")
        return results

# Usage example
if __name__ == "__main__":
    searcher = SemanticSearch()
    
    # Add some sample documents in different languages
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "El rápido zorro marrón salta sobre el perro perezoso",
        "Le rapide renard brun saute par-dessus le chien paresseux",
        "快速的棕色狐狸跳过懒惰的狗"
    ]
    searcher.add_documents(documents)
    
    # Perform searches in different languages
    queries = [
        "animal movement",
        "movimiento de animales",
        "mouvement des animaux",
        "动物运动"
    ]
    
    for query in queries:
        print(f"\nSearch Query: '{query}'")
        results = searcher.search(query, k=2)
        
        for result in results:
            print(f"Document: {result['document']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Language: {result['language']}")
            print()