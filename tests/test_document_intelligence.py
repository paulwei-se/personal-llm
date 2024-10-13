import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from document_intelligence import DocumentIntelligence

def test_document_intelligence():
    di = DocumentIntelligence()
    
    # Process and index sample documents
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_documents')
    for filename in os.listdir(sample_dir):
        if filename.endswith(('.jpg', '.png','.pdf')):
            file_path = os.path.join(sample_dir, filename)
            doc_id = di.process_and_index_document(file_path)
            print(f"Processed and indexed document: {filename} (ID: {doc_id})")
    
    # Test queries
    queries = [
        "computer science",
        "online purchase",
        "document processing",
        "GenAI"
    ]
    
    # Perform searches and print results
    for query in queries:
        print(f"\nSearch Query: '{query}'")
        results = di.search(query, k=3)
        
        for result in results:
            print(f"Document ID: {result['doc_id']}")
            print(f"Path: {result['path']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Snippet: {result['snippet']}")
            print()

if __name__ == "__main__":
    test_document_intelligence()
