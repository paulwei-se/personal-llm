from langchain.llms.base import LLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage
from langchain.schema import Document 

from typing import List, Dict, Any, Optional
import asyncio
import logging
from src.llm.llama_model import LlamaModel
from pydantic import Field, BaseModel
from pydantic.config import ConfigDict
import re


logger = logging.getLogger(__name__)

class CustomConversationMemory(BaseMemory):
   chat_history: List[tuple[str, str]] = []
   return_messages: bool = True
   max_history_turns: int = 10
   
   @property
   def memory_variables(self) -> List[str]:
       """Define the memory variables."""
       return ["chat_history"]

   def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        question = inputs["question"]
        # Clean the answer
        answer = outputs["answer"].strip()
        
        print("outputs:", outputs)
        
        # Extract actual answer if ANSWER: marker exists
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1].strip()
        sources = [
            f"Source: {doc.metadata['source']}"
            for doc in outputs.get("source_documents", [])
        ]

        print("saved_question:", question)
        print("saved_answer:", answer)
        print("saved_source:", sources)

        if sources:
            answer += f"\n\nReferences:\n{chr(10).join(sources)}"

        if len(self.chat_history) >= self.max_history_turns:
            self.chat_history = self.chat_history[2:]

        # Store as tuple
        self.chat_history.append((question, answer))


   def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
        # Get last few exchanges for context
        return {"chat_history": self.chat_history}

   def clear(self) -> None:
        self.chat_history = []

class LangChainLLamaWrapper(LLM, BaseModel):
    """Wrapper to make our LlamaModel compatible with LangChain"""
    
    # Define the model field properly
    llama_model: LlamaModel = Field(description="LlamaModel instance")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, llama_model: LlamaModel):
        super().__init__(llama_model=llama_model)
        self.llama_model = llama_model
        
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Async call to the model"""
        return await self.llama_model.generate(prompt=prompt)
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Synchronous call to generate text.
        This is called by LangChain's synchronous components.
        """
        try:
            # Get the current loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.llama_model.generate(prompt))
            
        if loop.is_running():
            # If we're already in an event loop, use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.llama_model.generate(prompt))
        else:
            # If loop exists but isn't running, use it
            return loop.run_until_complete(self.llama_model.generate(prompt))
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {"name": self.llama_model.model_name}
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "custom_llama"

class RAGPipeline:
    def __init__(self, llm: LlamaModel):
        self.llm = LangChainLLamaWrapper(llm)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Initialize vector store
        self.vector_store = None
        
        self.memory = CustomConversationMemory()
        
        self.qa_prompt = PromptTemplate.from_template(
            "Answer the following question based on the provided context.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Provide a clear and concise answer without any special formatting, links, or sections.\n"
            "Begin your answer with 'ANSWER:' and provide only the answer content:\n"
            "ANSWER:"
        )
        
        self.condense_question_prompt = PromptTemplate.from_template(
            "Given the conversation below, rephrase the follow-up question into a standalone question "
            "that captures the original intent.\n\n"
            "Chat History: {chat_history}\n"
            "Follow-up question: {question}\n\n"
            "Begin your rephrase with 'FOLLOWUP:' and provide only the follow-up question content:\n:"
            "FOLLOWUP:"
        )

    async def initialize_vector_store(self, documents: Dict[str, str]) -> None:
        """Initialize the vector store with documents"""
        try:
            # Process documents
            processed_docs = []
            for doc_id, content in documents.items():
                chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    # Create Document objects instead of dictionaries
                    processed_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": doc_id, "chunk_id": i}
                        )
                    )
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                processed_docs,
                self.embeddings,
                collection_name="document_store"
            )
            
            logger.info(f"Initialized vector store with {len(processed_docs)} chunks")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    async def process_query(
        self, 
        question: str,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query using the RAG pipeline"""
        try:
            if self.vector_store is None:
                return {"error": "Vector store not initialized"}
            
            # If doc_id is provided, filter for that document
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3, "filter": {"source": doc_id}} if doc_id else {"k": 3}
            )
            
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                condense_question_llm=self.llm,
                combine_docs_chain_kwargs={
                    "prompt": self.qa_prompt,
                    "document_variable_name": "context"
                },
                condense_question_prompt=self.condense_question_prompt,
                return_source_documents=True,
                get_chat_history=lambda h: h,
                verbose=True
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: qa_chain({
                    "question": question
                })
            )

            # Clean the answer by extracting only the content after "ANSWER:"
            answer = response["answer"]
            if "ANSWER:" in answer:
                answer = answer.split("ANSWER:")[-1].strip()

            return {
                "answer": answer,
                "sources": [
                    {"source": doc.metadata["source"]} 
                    for doc in response["source_documents"]
                ],
                "chat_history": self.memory.load_memory_variables({})["chat_history"]
            }

            # qa_chain = ConversationalRetrievalChain.from_llm(
            #     llm=self.llm,
            #     retriever=retriever,
            #     memory=self.memory,
            #     return_source_documents=True,
            #     combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            #     chain_type="stuff",
            #     return_generated_question=False,
            #     get_chat_history=lambda h: h,
            #     verbose=True
            # )

            # result = await asyncio.get_event_loop().run_in_executor(
            #     None,
            #     lambda: qa_chain({"question": question})
            # )
                
            # Format response aligning with CustomConversationMemory format
            # chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            # response = {
            #     "answer": result["answer"],
            #     "sources": [doc.metadata for doc in result["source_documents"]],
            #     "chat_history": chat_history,  # This now contains formatted messages with references
            #     "model_info": self.llm._identifying_params
            # }
            
            # return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {"error": str(e)}

    async def clear_memory(self) -> None:
        """Clear conversation memory"""
        self.memory.clear()

# Example usage
if __name__ == "__main__":
    async def test_rag():
        llm = LlamaModel()
        rag = RAGPipeline(llm)
        
        # Test documents
        documents = {
            "doc1": "LangChain is a framework for developing applications powered by language models.",
            "doc2": "RAG (Retrieval Augmented Generation) is a technique that combines retrieval and generation."
        }
        
        await rag.initialize_vector_store(documents)
        
        # Test query
        result = await rag.process_query("What is LangChain?")
        print(result)
        
        await rag.clear_memory()
    
    asyncio.run(test_rag())