import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
import asyncio
import unittest
from src.rag.rag_pipeline import RAGPipeline
from src.llm.llama_model import LlamaModel
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logging.basicConfig(level=logging.DEBUG)

@pytest_asyncio.fixture(scope="module")
async def rag_fixture():
    llm = LlamaModel()
    rag = RAGPipeline(llm=llm)
    documents = {
        "doc1": "LangChain is a framework for developing applications powered by language models.",
        "doc2": "RAG combines search with LLM generation to improve accuracy.",
        "doc3": "Vector databases store document embeddings for semantic search."
    }
    await rag.add_documents(documents, topic_id="langgraph")
    yield rag

@pytest.mark.asyncio
class TestRAGPipeline:
    async def test_basic_query(self, rag_fixture):
        """Test basic query without streaming."""
        query = "What is LangChain?"
        states = []
        # Collect all messages from the stream
        async for state in rag_fixture.chat(
            message=query,
            user_id="test_user",
            topic_id="langgraph"
        ):
            states.append(state)
            
        # Verify we got a response
        assert len(states) > 0
        assert isinstance(states[-1][-1], AIMessage)
        
        # Verify response content references source material
        response_content = states[-1][-1].content.lower()
        assert any([
            "framework" in response_content,
            "language model" in response_content,
            "langchain" in response_content
        ]), "Response should contain information from source documents"

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