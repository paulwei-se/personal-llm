import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search.semantic_search import SemanticSearch

def test_semantic_search():
    # Initialize the SemanticSearch instance
    searcher = SemanticSearch()
    
    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "In the beginning God created the heaven and the earth.",
        "Ask not what your country can do for you, ask what you can do for your country.",
        "The only thing we have to fear is fear itself.",
        "That's one small step for man, one giant leap for mankind."
    ]
    
    # Add documents to the search index
    searcher.add_documents(documents)
    
    # Test queries
    queries = [
        "animal movement",
        "philosophy",
        "famous quotes",
        "travel"
    ]
    
    # Perform searches and print results
    for query in queries:
        print(f"\nSearch Query: '{query}'")
        results = searcher.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Document: {result['document']}")
            print(f"   Score: {result['score']:.4f}")
        
        print()

if __name__ == "__main__":
    test_semantic_search()