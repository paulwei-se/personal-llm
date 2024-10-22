import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
import asyncio
from typing import Dict, List
import uuid
import logging
from src.rag.rag_pipeline import RAGPipeline
from src.llm.llama_model import LlamaModel
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.DEBUG)

# Test Data
TEST_DOCUMENTS = {
    "doc1": "LangChain is a framework for developing applications powered by language models. It provides tools for combining LLMs with other sources of computation or knowledge.",
    "doc2": "RAG (Retrieval Augmented Generation) combines search with LLM generation to improve accuracy. It helps ground LLM responses in source documents.",
    "doc3": "Vector databases store document embeddings for semantic search. They enable efficient similarity matching.",
    "doc4": "Python is a high-level programming language known for its simplicity and readability."
}

@pytest_asyncio.fixture(scope="module")
async def rag_fixture():
    """Create a RAGPipeline instance with test documents."""
    llm = LlamaModel()
    rag = RAGPipeline(llm=llm)
    
    # Add documents to different topics
    await rag.add_documents(
        {k: TEST_DOCUMENTS[k] for k in ["doc1", "doc2", "doc3"]}, 
        topic_id="ai_tech"
    )
    await rag.add_documents(
        {"doc4": TEST_DOCUMENTS["doc4"]}, 
        topic_id="programming"
    )
    
    yield rag

@pytest.fixture
def generate_test_ids():
    """Generate unique IDs for each test."""
    return {
        "user_id": f"test_user_{uuid.uuid4().hex[:8]}",
        "thread_id": f"test_thread_{uuid.uuid4().hex[:8]}"
    }

@pytest.mark.asyncio
class TestRAGPipeline:
    
    async def test_document_ingestion(self, rag_fixture, generate_test_ids):
        """Test adding new documents to the pipeline."""
        # Prepare test data
        new_docs = {
            "doc5": "Machine learning is a subset of artificial intelligence.",
            "doc6": "Deep learning uses neural networks with multiple layers."
        }
        topic_id = "ml_basics"
        
        # Add documents
        await rag_fixture.add_documents(new_docs, topic_id=topic_id)
        
        # Create test config
        config = {
            "configurable": {
                "thread_id": generate_test_ids["thread_id"],
                "checkpoint_ns": ""
            }
        }
        
        # Test querying the new documents
        responses = []
        async for state in rag_fixture.chat(
            message="What is machine learning?",
            user_id=generate_test_ids["user_id"],
            topic_id=topic_id,
            config=config
        ):
            responses.extend(state)
        
        # Verify response quality
        assert len(responses) > 0
        assert isinstance(responses[-1], AIMessage)
        assert "machine learning" in responses[-1].content.lower()
        assert "artificial intelligence" in responses[-1].content.lower()

    async def test_context_retrieval(self, rag_fixture, generate_test_ids):
        """Test if relevant context is retrieved for queries."""
        # Prepare test data
        query = "What is RAG and how does it work?"
        config = {
            "configurable": {
                "thread_id": generate_test_ids["thread_id"],
                "checkpoint_ns": ""
            }
        }
        
        # Test retrieval
        responses = []
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            responses = []
            async for state in rag_fixture.chat(
                message=query,
                user_id=generate_test_ids["user_id"],
                topic_id="ai_tech",
                config=config
            ):
                responses.extend(state)
            
            final_response = responses[-1].content.lower()
            if any([
                "retrieval" in final_response,
                "generation" in final_response,
                "search" in final_response,
                "rag" in final_response
            ]):
                success = True
                break
                
        assert success, f"Response should include key RAG concepts after {max_retries} attempts"

    async def test_multi_topic_isolation(self, rag_fixture, generate_test_ids):
        """Test if topics remain properly isolated."""
        # Query setup
        query = "Tell me about Python"
        base_thread_id = generate_test_ids["thread_id"]
        
        # Query in AI tech context
        ai_config = {
            "configurable": {
                "thread_id": f"{base_thread_id}_ai",
                "checkpoint_ns": ""
            }
        }
        
        ai_responses = []
        async for state in rag_fixture.chat(
            message=query,
            user_id=generate_test_ids["user_id"],
            topic_id="ai_tech",
            config=ai_config
        ):
            ai_responses.extend(state)
            
        # Query in programming context
        prog_config = {
            "configurable": {
                "thread_id": f"{base_thread_id}_prog",
                "checkpoint_ns": ""
            }
        }
        
        prog_responses = []
        async for state in rag_fixture.chat(
            message=query,
            user_id=generate_test_ids["user_id"],
            topic_id="programming",
            config=prog_config
        ):
            prog_responses.extend(state)
            
        # Verify responses
        ai_response = ai_responses[-1].content.lower()
        prog_response = prog_responses[-1].content.lower()
        
        # Check for programming-specific keywords
        programming_keywords = [
            "programming", "language", "code", "development",
            "scripting", "python", "interpreted"
        ]
        
        assert any(kw in prog_response for kw in programming_keywords), \
            "Programming response should contain programming-specific terms"
        assert prog_response != ai_response, \
            "Responses from different topics should be different"

    async def test_chat_history(self, rag_fixture, generate_test_ids):
        """Test chat history maintenance and retrieval."""
        user_id = generate_test_ids["user_id"]
        topic_id = "ai_tech"
        thread_id = f"{user_id}_{topic_id}"  # Match the format in get_chat_history
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ""
            }
        }
        
        # Test messages with expected topics in responses
        conversation = [
            ("What is LangChain?", ["framework", "language", "model"]),
            ("How does it relate to RAG?", ["retrieval", "generation", "search"]),
            ("Can you summarize what we discussed?", ["discuss", "mentioned", "summary"])
        ]
        
        # Send messages
        for message, expected_keywords in conversation:
            async for state in rag_fixture.chat(
                message=message,
                user_id=user_id,
                topic_id=topic_id,
                config=config
            ):
                # Just process the responses
                pass
        
        # Retrieve chat history using the pipeline's method
        history = rag_fixture.get_chat_history(
            user_id=user_id,
            topic_id=topic_id
        )
        
        # Verify history contents
        assert len(history) > 0, "Chat history should not be empty"
        assert len(history) >= len(conversation) * 2, \
            f"Expected at least {len(conversation) * 2} messages (including responses), got {len(history)}"
        
        # Verify message types alternate between human and AI
        for i, msg in enumerate(history):
            if i % 2 == 0:
                assert isinstance(msg, HumanMessage), f"Expected HumanMessage at position {i}"
            else:
                assert isinstance(msg, AIMessage), f"Expected AIMessage at position {i}"
        
        # Verify conversation flow and content
        human_messages = [msg.content for msg in history[::2]]  # Every other message starting from 0
        assert all(msg[0] in content for msg, content in zip(conversation, human_messages)), \
            "Original messages should be preserved in history"
        
        # Verify responses contain relevant keywords
        ai_messages = [msg.content.lower() for msg in history[1::2]]  # Every other message starting from 1
        for i, (msg, response) in enumerate(zip(conversation, ai_messages)):
            expected_keywords = msg[1]
            assert any(kw in response for kw in expected_keywords), \
                f"Response {i+1} should contain at least one of these keywords: {expected_keywords}"
        
        # Verify last message contains summary elements
        last_ai_message = ai_messages[-1]
        summary_keywords = ["discuss", "mentioned", "summary", "talked", "covered"]
        assert any(kw in last_ai_message for kw in summary_keywords), \
            "Final response should contain summary-related terms"

    async def test_concurrent_queries(self, rag_fixture, generate_test_ids):
        """Test handling of concurrent queries."""
        async def make_query(message: str, user_suffix: str) -> List[str]:
            config = {
                "configurable": {
                    "thread_id": f"{generate_test_ids['thread_id']}_{user_suffix}",
                    "checkpoint_ns": ""
                }
            }
            
            responses = []
            async for state in rag_fixture.chat(
                message=message,
                user_id=f"{generate_test_ids['user_id']}_{user_suffix}",
                topic_id="ai_tech",
                config=config
            ):
                responses.extend(state)
            return responses
        
        # Run multiple queries concurrently
        queries = [
            ("What is LangChain?", "user1", ["framework", "language", "model"]),
            ("What is RAG?", "user2", ["retrieval", "generation", "search"]),
            ("Tell me about vector databases", "user3", ["vector", "database", "embedding", "search"])
        ]
        
        tasks = [make_query(q[0], q[1]) for q in queries]
        results = await asyncio.gather(*tasks)
    
        # Verify each query got a valid response
        for i, responses in enumerate(results):
            assert len(responses) > 0
            assert isinstance(responses[-1], AIMessage)
            # Print the actual response for debugging
            final_response = responses[-1].content.lower()
            print(f"\nQuery: {queries[i][0]}")
            print(f"Response: {final_response}")
            print(f"Expected keywords: {queries[i][2]}")
            print(f"Found keywords: {[kw for kw in queries[i][2] if kw in final_response]}")
            
            assert any(kw in final_response for kw in queries[i][2]), \
                f"Response should contain at least one of these keywords: {queries[i][2]}\nActual response: {final_response}"

    @pytest.mark.parametrize("error_case", [
        ("nonexistent_topic", "What is AI?"),
        ("ai_tech", ""),  # Empty query
        ("ai_tech", "x" * 1000),  # Very long query
    ])
    async def test_error_handling(self, rag_fixture, generate_test_ids, error_case):
        """Test error handling for various edge cases."""
        topic_id, query = error_case
        
        # Create unique config for this error test
        config = {
            "configurable": {
                "thread_id": f"{generate_test_ids['thread_id']}_{topic_id}_{len(query)}",
                "checkpoint_ns": ""
            }
        }
        
        try:
            responses = []
            async for state in rag_fixture.chat(
                message=query,
                user_id=f"{generate_test_ids['user_id']}_{topic_id}",
                topic_id=topic_id,
                config=config
            ):
                responses.extend(state)
            
            # For cases where we expect success
            if topic_id == "ai_tech" and len(query) < 1000:
                assert len(responses) > 0
                assert isinstance(responses[-1], AIMessage)
                
        except Exception as e:
            # For cases where we expect an error
            if topic_id == "nonexistent_topic":
                assert "topic not found" in str(e).lower()
            elif not query:
                assert "empty query" in str(e).lower()
            elif len(query) >= 1000:
                assert "query length" in str(e).lower()

if __name__ == "__main__":
    asyncio.run(pytest.main(["-v", __file__]))