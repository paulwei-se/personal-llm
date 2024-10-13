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
        if filename.endswith(('.jpg', '.png', '.pdf')):
            file_path = os.path.join(sample_dir, filename)
            doc_id = di.process_and_index_document(file_path)
            print(f"Processed and indexed document: {filename} (ID: {doc_id})")
    
    # Test queries in different languages
    queries = [
        "computer science",
        "inteligencia artificial",
        "apprentissage automatique",
        "数据科学"
    ]
    
    # Perform searches and print results
    for query in queries:
        print(f"\nSearch Query: '{query}'")
        results = di.search(query, k=3)
        
        if results:
            for result in results:
                print(f"Document ID: {result['doc_id']}")
                print(f"Path: {result['path']}")
                print(f"Score: {result['score']:.4f}")
                print(f"Language: {result['language']}")
                print(f"Snippet: {result['snippet']}")
                print()
        else:
            print("No results found.")
    
    # Test question answering
    if di.documents:
        first_doc_id = list(di.documents.keys())[0]
        question = "What is the main topic of this document?"
        answer = di.answer_question(question, first_doc_id)
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer['answer']}")
        print(f"Confidence: {answer['score']:.4f}")
    
    # Test summarization
    if di.documents:
        summary = di.summarize_document(first_doc_id)
        print(f"\nSummary of document {first_doc_id}:")
        print(summary)

    # Test error handling
    print("\nTesting error handling:")
    
    # Test with non-existent document
    print("Searching with non-existent document:")
    results = di.search("This document doesn't exist", k=1)
    print("Results:", results)
    
    # Test question answering with invalid document ID
    print("\nQuestion answering with invalid document ID:")
    answer = di.answer_question("What is this?", "invalid_id")
    print("Answer:", answer)
    
    # Test summarization with invalid document ID
    print("\nSummarizing with invalid document ID:")
    summary = di.summarize_document("invalid_id")
    print("Summary:", summary)

if __name__ == "__main__":
    test_document_intelligence()