import os
import logging
from document_processing.processor import DocumentProcessor
from search.semantic_search import SemanticSearch
from qa_summary.qa_summarizer import QASummarizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentIntelligence:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.semantic_search = SemanticSearch()
        self.qa_summarizer = QASummarizer()
        self.documents = {}  # Store processed documents with their IDs
        self.logger = logging.getLogger(__name__)

    def process_and_index_document(self, document_path):
        """Process a document and add it to the semantic search index."""
        try:
            doc_id = str(len(self.documents) + 1)
            processed_doc = self.document_processor.process_document(document_path)
            
            self.documents[doc_id] = {
                'path': document_path,
                'content': processed_doc['text'],
                'layout': processed_doc['layout'],
                'language': self.semantic_search.detect_language(processed_doc['text'])
            }
            
            self.semantic_search.add_documents([processed_doc['text']])
            
            self.logger.info(f"Successfully processed and indexed document: {document_path} (ID: {doc_id}, Language: {self.documents[doc_id]['language']})")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            return None

    def search(self, query, k=5):
        """Perform a semantic search on the indexed documents."""
        try:
            results = self.semantic_search.search(query, k)
            
            enriched_results = []
            for result in results:
                doc_id = next((id for id, doc in self.documents.items() if doc['content'] == result['document']), None)
                if doc_id is not None:
                    enriched_results.append({
                        'doc_id': doc_id,
                        'path': self.documents[doc_id]['path'],
                        'score': result['score'],
                        'snippet': result['document'][:200] + '...',
                        'language': result['language']
                    })
            
            self.logger.info(f"Search query '{query}' returned {len(enriched_results)} results")
            return enriched_results
        except Exception as e:
            self.logger.error(f"Error performing search with query '{query}': {str(e)}")
            return []

    def answer_question(self, question, doc_id):
        """Answer a question based on a specific document."""
        try:
            if doc_id not in self.documents:
                self.logger.warning(f"Document ID {doc_id} not found")
                return {"answer": "Document not found", "score": 0.0}
            
            document = self.documents[doc_id]['content']
            answer = self.qa_summarizer.answer_question(document, question)
            self.logger.info(f"Answered question for document {doc_id} in {self.documents[doc_id]['language']}")
            return answer
        except Exception as e:
            self.logger.error(f"Error answering question for document {doc_id}: {str(e)}")
            return {"answer": "Unable to answer the question", "score": 0.0}

    def summarize_document(self, doc_id, max_length=150, min_length=50):
        """Generate a summary for a specific document."""
        try:
            if doc_id not in self.documents:
                self.logger.warning(f"Document ID {doc_id} not found")
                return "Document not found"
            
            document = self.documents[doc_id]['content']
            summary = self.qa_summarizer.summarize_text(document, max_length, min_length)
            self.logger.info(f"Generated summary for document {doc_id} in {self.documents[doc_id]['language']}")
            return summary
        except Exception as e:
            self.logger.error(f"Error summarizing document {doc_id}: {str(e)}")
            return "Unable to generate summary"

# Usage example
if __name__ == "__main__":
    di = DocumentIntelligence()
    
    # Process and index sample documents in different languages
    sample_docs = [
        "path/to/english_document.pdf",
        "path/to/spanish_document.pdf",
        "path/to/french_document.pdf",
        "path/to/chinese_document.pdf"
    ]
    
    for doc_path in sample_docs:
        doc_id = di.process_and_index_document(doc_path)
        if doc_id:
            print(f"Processed document: {doc_path} (ID: {doc_id}, Language: {di.documents[doc_id]['language']})")
    
    # Perform searches in different languages
    queries = [
        "technology advancements",
        "avances tecnológicos",
        "progrès technologiques",
        "技术进步"
    ]
    
    for query in queries:
        print(f"\nSearch Query: '{query}'")
        results = di.search(query, k=2)
        for result in results:
            print(f"Document ID: {result['doc_id']}")
            print(f"Path: {result['path']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Language: {result['language']}")
            print(f"Snippet: {result['snippet']}")
            print()
    
    # Answer a question (assuming the first document is in English)
    if di.documents:
        first_doc_id = next(iter(di.documents))
        answer = di.answer_question("What are the main topics discussed in this document?", first_doc_id)
        print("Answer:", answer)
    
        # Generate a summary
        summary = di.summarize_document(first_doc_id)
        print("Summary:", summary)
    
    # Test with invalid document ID
    invalid_answer = di.answer_question("What is this?", "invalid_id")
    print("Answer for invalid ID:", invalid_answer)
    
    invalid_summary = di.summarize_document("invalid_id")
    print("Summary for invalid ID:", invalid_summary)