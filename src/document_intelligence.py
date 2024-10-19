import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.document_processing.processor import DocumentProcessor, chunk_document
from src.search.semantic_search import SemanticSearch
from src.qa_summary.qa_summarizer import QASummarizer
from src.ethical_ai.bias_detector import BiasDetector
from src.ethical_ai.explainer import AIExplainer
from src.llm.llama_model import LlamaModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentIntelligence:
    def __init__(self, max_workers=4):
        self.document_processor = DocumentProcessor()
        self.semantic_search = SemanticSearch()
        # self.qa_summarizer = QASummarizer()
        self.bias_detector = BiasDetector()
        self.ai_explainer = AIExplainer()
        self.documents = {}  # Store processed documents with their IDs
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.llm = LlamaModel()  # Initialize Llama model

    async def process_and_index_document(self, document_path):
        doc_id = str(len(self.documents) + 1)
        try:
            text = await asyncio.to_thread(self.document_processor.extract_text, document_path)
            chunks = await asyncio.to_thread(chunk_document, text)
            layout = await asyncio.to_thread(self.document_processor.analyze_layout, document_path)
            
            self.documents[doc_id] = {
                'path': document_path,
                'content': text,
                'chunks': chunks,
                'layout': layout,
                'language': await asyncio.to_thread(self.semantic_search.detect_language, text)
            }
            
            await self.semantic_search.add_documents(chunks, doc_id)
            return doc_id
        except Exception as e:
            self.logger.error(f"Error processing document {document_path}: {str(e)}")
            return None
    
    def bulk_process_and_index(self, document_paths):
        """Process and index multiple documents concurrently."""
        futures = []
        for path in document_paths:
            future = self.executor.submit(self.process_and_index_document, path)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                doc_id = future.result()
                results.append(doc_id)
            except Exception as e:
                self.logger.error(f"Error in bulk processing: {str(e)}")

        return results
    
    async def remove_document(self, doc_id):
        if doc_id in self.documents:
            await self.semantic_search.remove_document(doc_id)
            del self.documents[doc_id]
            return True
        return False

    async def search(self, query, k=5):
        results = await self.semantic_search.search(query, k)
        return [
            {
                'doc_id': result['doc_id'],
                'chunk_id': result['chunk_id'],
                'path': self.documents[result['doc_id']]['path'],
                'score': result['score'],
                'snippet': result['chunk'][:200] + '...',
                'language': result['language']
            }
            for result in results
        ]

    async def answer_question(self, question, doc_id):
        if doc_id not in self.documents:
            return {"answer": "Document not found", "score": 0.0}
        
        relevant_chunks = await self.semantic_search.search(question, k=3, doc_id=doc_id)
        context = " ".join([chunk['chunk'] for chunk in relevant_chunks])

        prompt = f"""Context: {context}

        Question: {question}

        Please provide a concise and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, please say so.

        Answer:"""

        # Generate answer using Llama
        answer = await self.llm.generate(prompt, max_length=256)

        # Get explanation using existing explainer
        explanation = await self.ai_explainer.explain_answer(context, question, answer)

        # answer = await self.qa_summarizer.answer_question(context, question, doc_id)
        # explanation = await self.ai_explainer.explain_answer(context, question, answer['answer'])
        
        return {**answer, 'explanation': explanation}

    async def summarize_document(self, doc_id, max_length=150, min_length=50):
        if doc_id not in self.documents:
            return "Document not found"
        
        document = self.documents[doc_id]['content']
        # Create a summarization prompt
        prompt = f"""Please provide a concise summary of the following text:

        {document}

        Summary:"""

        summary = await self.llm.generate(
            prompt,
            max_length=max_length,
            temperature=0.7
        )
        
        # summary = await self.qa_summarizer.summarize_text(document, max_length, min_length)
        return summary

    async def detect_bias(self, doc_id):
        if doc_id not in self.documents:
            return {"error": "Document not found"}
        
        document = self.documents[doc_id]['content']
        sentences = document.split('.')  # Simple sentence splitting
        sentiment_bias = await self.bias_detector.detect_sentiment_bias(sentences)
        toxicity = await self.bias_detector.detect_toxicity(sentences)
        return {'sentiment_bias': sentiment_bias, 'toxicity': toxicity}

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