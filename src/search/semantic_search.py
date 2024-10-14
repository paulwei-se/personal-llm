import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        self.index = None
        self.documents = {}
        self.document_languages = {}
        self.embedding_size = None
        self.total_chunks = 0
        self.IVF_THRESHOLD = 10000  # Minimum number of chunks before using IVF

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            self.logger.warning(f"Could not detect language. Defaulting to English.")
            return 'en'

    async def add_documents(self, chunks, doc_id):
        embeddings = await asyncio.to_thread(self.model.encode, chunks)
        
        if self.embedding_size is None:
            self.embedding_size = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embedding_size)

        start_id = self.total_chunks
        self.index.add(embeddings)
        self.documents[doc_id] = chunks
        self.document_languages[doc_id] = detect(chunks[0])
        self.total_chunks += len(chunks)

        # Check if we need to switch to IVF index
        if self.total_chunks >= self.IVF_THRESHOLD and not isinstance(self.index, faiss.IndexIVFFlat):
            self._switch_to_ivf()

    def _switch_to_ivf(self):
        print("Switching to IVF index...")
        nlist = min(int(np.sqrt(self.total_chunks)), 100)  # Number of clusters, capped at 100
        quantizer = faiss.IndexFlatL2(self.embedding_size)
        new_index = faiss.IndexIVFFlat(quantizer, self.embedding_size, nlist)
        
        # Train on all existing vectors
        all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
        new_index.train(all_vectors)
        new_index.add(all_vectors)
        
        self.index = new_index

    async def search(self, query, k=5, doc_id=None):
        query_embedding = await asyncio.to_thread(self.model.encode, [query])
        
        if doc_id:
            # Search within a specific document
            doc_chunks = self.documents.get(doc_id, [])
            if not doc_chunks:
                return []
            
            doc_embeddings = await asyncio.to_thread(self.model.encode, doc_chunks)
            doc_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
            doc_index.add(doc_embeddings)
            
            distances, indices = doc_index.search(query_embedding, min(k, len(doc_chunks)))
            
            results = [
                {
                    'doc_id': doc_id,
                    'chunk_id': int(idx),
                    'chunk': doc_chunks[int(idx)],
                    'score': float(1 - dist / 2),
                    'language': self.document_languages[doc_id]
                }
                for idx, dist in zip(indices[0], distances[0])
            ]
        else:
            # Search across all documents
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i in range(len(indices[0])):
                idx = int(indices[0][i])
                distance = float(distances[0][i])
                
                for doc_id, chunks in self.documents.items():
                    if idx < len(chunks):
                        results.append({
                            'doc_id': doc_id,
                            'chunk_id': idx,
                            'chunk': chunks[idx],
                            'score': float(1 - distance / 2),
                            'language': self.document_languages[doc_id]
                        })
                        break
                    idx -= len(chunks)

                if len(results) == k:
                    break

        return results

    def _get_doc_and_chunk_id(self, global_idx):
        current_count = 0
        for doc_id, chunks in self.documents.items():
            if current_count + len(chunks) > global_idx:
                return doc_id, global_idx - current_count
            current_count += len(chunks)
        return None, None  # Should never reach here if index is consistent

    async def remove_document(self, doc_id):
        if doc_id in self.documents:
            num_chunks = len(self.documents[doc_id])
            self.total_chunks -= num_chunks
            del self.documents[doc_id]
            del self.document_languages[doc_id]
            
            # Rebuild index
            all_chunks = [chunk for chunks in self.documents.values() for chunk in chunks]
            all_embeddings = await asyncio.to_thread(self.model.encode, all_chunks)
            
            if self.total_chunks >= self.IVF_THRESHOLD:
                self._create_ivf_index(all_embeddings)
            else:
                self.index = faiss.IndexFlatL2(self.embedding_size)
                self.index.add(all_embeddings)

            return True
        return False

    def _create_ivf_index(self, vectors):
        nlist = min(int(np.sqrt(len(vectors))), 100)  # Number of clusters, capped at 100
        quantizer = faiss.IndexFlatL2(self.embedding_size)
        new_index = faiss.IndexIVFFlat(quantizer, self.embedding_size, nlist)
        new_index.train(vectors)
        new_index.add(vectors)
        self.index = new_index

# Usage example
if __name__ == "__main__":
    async def test():
        searcher = SemanticSearch()
        
        # Add some sample documents
        docs = [
            "The quick brown fox jumps over the lazy dog",
            "Le rapide renard brun saute par-dessus le chien paresseux",
            "快速的棕色狐狸跳过懒惰的狗"
        ]
        for i, doc in enumerate(docs):
            await searcher.add_documents([doc], f"doc_{i}")
        
        # Perform a search
        results = await searcher.search("fox jumping", k=2)
        for result in results:
            print(f"Document: {result['doc_id']}")
            print(f"Chunk: {result['chunk']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Language: {result['language']}")
            print()

    asyncio.run(test())