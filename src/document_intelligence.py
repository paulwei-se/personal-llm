import os
import logging
from document_processing.processor import DocumentProcessor
from search.semantic_search import SemanticSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentIntelligence:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.semantic_search = SemanticSearch()
        self.documents = {}  # Store processed documents with their IDs
        self.logger = logging.getLogger(__name__)

    def process_and_index_document(self, document_path):
        """Process a document and add it to the semantic search index."""
        try:
            # Generate a simple document ID
            doc_id = str(len(self.documents) + 1)
            
            # Process the document
            processed_doc = self.document_processor.process_document(document_path)
            
            # Store the processed document
            self.documents[doc_id] = {
                'path': document_path,
                'content': processed_doc['text'],
                'layout': processed_doc['layout']
            }
            
            # Add the document text to the semantic search index
            self.semantic_search.add_documents([processed_doc['text']])
            
            self.logger.info(f"Successfully processed and indexed document: {document_path} (ID: {doc_id})")
            return doc_id
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            raise

    def search(self, query, k=5):
        """Perform a semantic search on the indexed documents."""
        try:
            results = self.semantic_search.search(query, k)
            
            # Enrich the results with document metadata
            enriched_results = []
            for result in results:
                doc_id = next(id for id, doc in self.documents.items() if doc['content'] == result['document'])
                enriched_results.append({
                    'doc_id': doc_id,
                    'path': self.documents[doc_id]['path'],
                    'score': result['score'],
                    'snippet': result['document'][:200] + '...'  # First 200 characters as a snippet
                })
            
            self.logger.info(f"Search query '{query}' returned {len(enriched_results)} results")
            return enriched_results
        except Exception as e:
            self.logger.error(f"Error performing search with query '{query}': {str(e)}")
            raise

# Usage example
if __name__ == "__main__":
    di = DocumentIntelligence()
    
    # Process and index some sample documents
    sample_dir = "path/to/sample/documents"
    for filename in os.listdir(sample_dir):
        if filename.endswith(('.jpg', '.png', '.pdf')):  # Add support for PDFs later
            file_path = os.path.join(sample_dir, filename)
            doc_id = di.process_and_index_document(file_path)
            print(f"Processed and indexed document: {filename} (ID: {doc_id})")
    
    # Perform a search
    query = "example query"
    results = di.search(query)
    
    print(f"\nSearch results for query: '{query}'")
    for result in results:
        print(f"Document ID: {result['doc_id']}")
        print(f"Path: {result['path']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Snippet: {result['snippet']}")
        print()
