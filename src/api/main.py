# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import asyncio
import time
import os
import sys
from src.rag.rag_pipeline import RAGPipeline
from src.llm.llama_model import LlamaModel
from src.document_processing.processor import DocumentProcessor
import logging
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

app = FastAPI()

# Initialize components with debug info
logger.info("Initializing LLM...")
llm = LlamaModel()
logger.info("Initializing RAG Pipeline...")
rag = RAGPipeline(llm=llm)
logger.info("Initializing Document Processor...")
doc_processor = DocumentProcessor()

# Create temp directory
if not os.path.exists("temp_uploads"):
    os.makedirs("temp_uploads")
    logger.debug("Created temp_uploads directory")

document_indices: Dict[str, Dict[str, Set[str]]] = {}  # topic_id -> {doc_id -> set of chunk_indices}

class ChatMessage(BaseModel):
    message: str
    user_id: str
    topic_id: str

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    topic_id: str = "default",
    doc_id: str = None
):
    try:
        # Generate doc_id if not provided
        if not doc_id:
            base_name, extension = os.path.splitext(file.filename)
            doc_id = f"{base_name}_{int(time.time() * 1000)}{extension}"
        
        temp_path = f"temp_uploads/temp_{doc_id}"
        os.makedirs("temp_uploads", exist_ok=True)
        
        try:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process document
            doc_content = doc_processor.extract_text(temp_path)['text']
            
            # Initialize tracking for this topic if needed
            if topic_id not in document_indices:
                document_indices[topic_id] = {}
            
            # Add to RAG pipeline
            chunks = await rag.add_documents(
                {doc_id: doc_content},
                topic_id=topic_id,
                metadata={
                    "original_filename": file.filename,
                    "doc_id": doc_id,
                    "topic_id": topic_id,
                    "upload_time": time.time()
                }
            )
            
            # Track the UUIDs for this document
            if chunks:
                # Get all UUIDs for this document from the vector store
                vector_store = rag.vector_stores[topic_id]
                document_uuids = set()
                for uuid, doc in vector_store.docstore._dict.items():
                    if doc.metadata.get("doc_id") == doc_id:
                        document_uuids.add(uuid)
                
                document_indices[topic_id][doc_id] = document_uuids
                
                logger.debug(
                    f"Added document {doc_id} with UUIDs "
                    f"{document_indices[topic_id][doc_id]}"
                )
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "original_filename": file.filename,
                "topic_id": topic_id,
                "chunk_count": len(chunks) if chunks else 0
            }
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    
@app.get("/topics")
async def get_topics():
    logger.debug("Getting list of topics")
    topics = list(rag.vector_stores.keys())
    logger.debug(f"Found topics: {topics}")
    return {"topics": topics}

@app.get("/documents/{topic_id}")
async def get_documents(topic_id: str):
    """Get all documents for a specific topic"""
    try:
        if topic_id in rag.vector_stores:
            docs = list(rag.vector_stores[topic_id].docstore._dict.values())
            unique_docs = {}
            
            # Use tracked document indices to get unique documents
            for doc_id in document_indices.get(topic_id, {}):
                # Get first chunk of document for metadata
                for doc in docs:
                    if doc.metadata.get("doc_id") == doc_id:
                        unique_docs[doc_id] = {
                            "id": doc_id,
                            "name": doc.metadata.get("original_filename", doc_id),
                            "topic_id": doc.metadata.get("topic_id", topic_id),
                            "upload_time": doc.metadata.get("upload_time", None)
                        }
                        break
            
            return {"documents": list(unique_docs.values())}
        return {"documents": []}
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{topic_id}/{doc_id}")
async def delete_document(topic_id: str, doc_id: str):
    try:
        logger.debug(f"Delete request - Topic: {topic_id}, Document: {doc_id}")
        
        # Log current state for debugging
        if topic_id in document_indices:
            logger.debug(f"Document indices for topic {topic_id}: {document_indices[topic_id]}")
        if topic_id in rag.vector_stores:
            store = rag.vector_stores[topic_id]
            logger.debug(f"Vector store indices: {store.index_to_docstore_id}")
            logger.debug(f"Docstore contents: {[(k, v.metadata) for k, v in store.docstore._dict.items()]}")
        
        try:
            # Use RAG pipeline to remove document
            await rag.remove_documents([doc_id], topic_id)
            
            # Update tracking if successful
            if topic_id in document_indices:
                if doc_id in document_indices[topic_id]:
                    del document_indices[topic_id][doc_id]
                if not document_indices[topic_id]:
                    del document_indices[topic_id]
                    
            return {"status": "success"}
            
        except ValueError as e:
            logger.error(f"Error details: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(message: ChatMessage):
    logger.debug(f"Chat request - User: {message.user_id}, Topic: {message.topic_id}")
    logger.debug(f"Message: {message.message}")
    try:
        async def generate_stream():
            async for messages in rag.chat(
                message=message.message,
                user_id=message.user_id,
                topic_id=message.topic_id
            ):
                # Ensure we're sending a properly formatted JSON string
                if messages:
                    latest_message = messages[-1]  # Get the latest message
                    yield f"data: {json.dumps({'content': latest_message.content})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{user_id}/{topic_id}")
async def get_chat_history(user_id: str, topic_id: str):
    logger.debug(f"Getting chat history - User: {user_id}, Topic: {topic_id}")
    try:
        history = rag.get_chat_history(user_id, topic_id)
        logger.debug(f"Found {len(history)} messages in history")
        formatted_history = [
            {"role": "human" if msg.type == "human" else "assistant", 
             "content": msg.content}
            for msg in history
        ]
        return {"history": formatted_history}
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

