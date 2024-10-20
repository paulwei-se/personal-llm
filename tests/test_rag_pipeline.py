import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
from src.rag.rag_pipeline import RAGPipeline
from src.llm.llama_model import LlamaModel
import logging

logging.basicConfig(level=logging.INFO)

@pytest_asyncio.fixture(scope="module")
async def rag_fixture():
    llm = LlamaModel()
    rag = RAGPipeline(llm=llm)
    documents = {
        "doc1": "LangChain is a framework for developing applications powered by language models.",
        "doc2": "RAG combines search with LLM generation to improve accuracy.",
        "doc3": "Vector databases store document embeddings for semantic search."
    }
    await rag.initialize_vector_store(documents)
    yield rag
    await rag.clear_memory()

@pytest.mark.asyncio
class TestRAGPipeline:
    # async def test_vector_store_initialization(self, rag_fixture):
    #     assert rag_fixture.vector_store is not None

    # async def test_single_document_query(self, rag_fixture):
    #     result = await rag_fixture.process_query("What is LangChain?", doc_id="doc1")
    #     print("RAG instance:", rag_fixture)
    #     print("Result:", result) 
    #     assert "framework" in result["answer"].lower()
    #     assert result["sources"][0]["source"] == "doc1"

    async def test_conversation_memory(self, rag_fixture):
        result1 = await rag_fixture.process_query("What is LangChain?")
        print("RAG instance:", rag_fixture)
        print("Result:", result1) 
        print("\nFirst Query Result:", result1)
        
        result2 = await rag_fixture.process_query("Can you elaborate on that?")
        print("\nFollow-up Query Result:", result2)
        
        assert len(result2["chat_history"]) > len(result1["chat_history"])
        assert "framework" in result1["answer"].lower()

    # async def test_basic_query(self, rag_fixture):
    #     async with rag_fixture() as rag:
    #         result = await rag.process_query("What is LangChain?")
    #         assert "error" not in result
    #         assert "answer" in result
    #         assert len(result["sources"]) > 0
    #         assert any("doc1" in source["source"] for source in result["sources"])

    # async def test_filtered_query(self, rag_fixture):
    #     async with rag_fixture() as rag:
    #         result = await rag.process_query("What is RAG?", doc_id="doc2")
    #         assert "error" not in result
    #         assert "answer" in result
    #         assert all("doc2" in source["source"] for source in result["sources"])

    # async def test_memory_management(self, rag_fixture):
    #     async with rag_fixture() as rag:
    #         result1 = await rag.process_query("What is LangChain?")
    #         assert len(rag.memory.chat_memory.messages) > 0
    #         await rag.clear_memory()
    #         assert len(rag.memory.chat_memory.messages) == 0

if __name__ == "__main__":
    asyncio.run(pytest.main(["-v", __file__]))