from typing import Annotated, Dict, List, Optional
from typing_extensions import TypedDict
import asyncio
import logging
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.llm.llama_model import LlamaModel

logger = logging.getLogger(__name__)

# Define our state schema
class State(TypedDict):
    """The state of our RAG application."""
    messages: Annotated[list, add_messages]  # Chat history
    context: Optional[List[Document]]        # Retrieved documents
    user_id: str                            # User identifier
    topic_id: Optional[str]                 # Topic/conversation identifier

class RAGPipeline:
    def __init__(self, llm):
        # Initialize core components
        self.llm = llm
        self.memory = MemorySaver()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.vector_stores: Dict[str, FAISS] = {}  # Topic-specific vector stores
        
        # Initialize the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the RAG processing graph."""
        graph_builder = StateGraph(State)
        
         # Add nodes - using async functions
        graph_builder.add_node("retrieve", self._retrieve_context)
        graph_builder.add_node("generate", self._generate_response)
        
        # Define edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", END)
        
        return graph_builder.compile(checkpointer=self.memory)

    async def _retrieve_context(self, state: State) -> Dict:
        """Retrieve relevant context based on the latest message."""
        if not state["messages"]:
            return {"context": None}
            
        latest_message = state["messages"][-1]
        if isinstance(latest_message, HumanMessage):
            query = latest_message.content
        else:
            return {"context": state.get("context")}
            
        vector_store = self.vector_stores.get(state["topic_id"])
        if not vector_store:
            return {"context": None}
            
        # Use ainvoke for async vector search
        docs = await vector_store.asimilarity_search(query, k=3)
    
        return {"context": docs}

    async def _generate_response(self, state: State) -> Dict:
        """Generate a response using the LLM with retrieved context."""
        context = state.get("context", [])
        messages = state.get("messages", [])
        
        if not messages:
            return {"messages": [], "context": context}
        
        context_str = "Use your own knowledge"
        if context:
            # Format context in a more structured way
            context_str = "\n".join(
                f"[Source {i+1}]: {doc.page_content}"
                for i, doc in enumerate(context)
            )
            
        prompt = f"""### Instruction:
You are a helpful AI assistant. Use ONLY the provided CONTEXT to answer the QUESTION accurately. 
If the context doesn't contain enough information, say so instead of making up information.
Be specific and include key terms from the context in your response.

### CONTEXT:
{context_str}

### QUESTION:
{messages[-1].content}

### RESPONSE:"""

        # Generate response
        response = await self.llm.generate(prompt)
        
        return {
            "messages": messages + [AIMessage(content=response)],
            "context": context
        }

    async def add_documents(self, documents: List[str], topic_id: str, metadata: Dict = None):
        """Process and index documents for a specific topic."""
        if not documents:
            raise ValueError("No documents provided")
        if not topic_id:
            raise ValueError("Topic ID must be provided")
        try:
            processed_docs = []
            chunks = []
            for doc_id, content in documents.items():
                text_chunks = self.text_splitter.split_text(content)
                chunks.extend(text_chunks)
                processed_docs.extend([
                    Document(
                        page_content=chunk,
                        metadata={
                            **(metadata or {}),
                            "chunk_index": len(processed_docs) + i,
                        }
                    )
                    for i, chunk in enumerate(text_chunks)
                ])

            if topic_id not in self.vector_stores:
                self.vector_stores[topic_id] = FAISS.from_documents(
                    processed_docs,
                    self.embeddings
                )
            else:
                self.vector_stores[topic_id].add_documents(processed_docs)
                
            logger.info(f"Added {len(processed_docs)} chunks to topic {topic_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def remove_documents(self, doc_ids: List[str], topic_id: str) -> bool:
        """Remove documents from the vector store for a specific topic."""
        if topic_id not in self.vector_stores:
            raise ValueError(f"Topic {topic_id} not found")
            
        try:
            vector_store = self.vector_stores[topic_id]
            
            # Find all UUIDs that match the given doc_ids
            uuids_to_remove = []
            for uuid, doc in vector_store.docstore._dict.items():
                if doc.metadata.get("doc_id") in doc_ids:
                    uuids_to_remove.append(uuid)
                    
            if not uuids_to_remove:
                raise ValueError(f"No documents found with IDs: {doc_ids}")
            
            logger.debug(f"Removing UUIDs: {uuids_to_remove} for doc_ids: {doc_ids}")
            
            # Remove documents from vector store
            await vector_store.adelete(uuids_to_remove)
                
            # Clean up empty topics
            if len(vector_store.docstore._dict) == 0:
                del self.vector_stores[topic_id]
                    
            return True
            
        except Exception as e:
            logger.error(f"Error removing documents: {str(e)}")
            raise

    async def chat(
        self,
        message: str,
        user_id: str,
        topic_id: str,
        config: Optional[Dict] = None
    ):
        """Process a chat message and return the response."""
        # Input validation
        if not message.strip():
            raise ValueError("Empty query not allowed")
        if topic_id not in self.vector_stores:
            raise ValueError(f"Topic {topic_id} not found")
        
        thread_id = f"{user_id}_{topic_id}"
    
        if config is None:
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ""
                }
            }
        
        # Get existing state or create new one
        try:
            current_state = self.graph.get_state(config)
            existing_messages = current_state.values.get("messages", [])
        except:
            existing_messages = []

        inputs = {
            "messages": existing_messages + [HumanMessage(content=message)],
            "user_id": user_id,
            "topic_id": topic_id,
            "context": None
        }

        async for output in self.graph.astream(
            inputs,
            config,
            stream_mode="updates"
        ):
            for node_name, value in output.items():
                logger.debug(f"Node {node_name} output: {value}")
                if "messages" in value and value["messages"]:
                    yield value["messages"]


    def get_chat_history(self, user_id: str, topic_id: str) -> List[BaseMessage]:
        """Retrieve chat history for a specific user and topic."""
        config = {
            "configurable": {
                "thread_id": f"{user_id}_{topic_id}",
                "checkpoint_ns": ""
            }
        }
        
        try:
            state = self.graph.get_state(config)
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []