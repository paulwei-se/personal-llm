import streamlit as st
import requests
import json
import time
import uuid
import os
from typing import Dict, List

API_URL = "http://localhost:8000"

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "current_topic" not in st.session_state:
    st.session_state.current_topic = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = {}  # Dict to store messages by topic
if "topics" not in st.session_state:
    # Only load topics from backend during initialization
    try:
        response = requests.get(f"{API_URL}/topics")
        if response.status_code == 200:
            st.session_state.topics = set(response.json()["topics"])
        else:
            st.session_state.topics = set()
    except Exception as e:
        st.error(f"Error loading initial topics: {str(e)}")
        st.session_state.topics = set()

def generate_unique_doc_id(filename: str) -> str:
    """Generate a unique document ID while preserving the file extension"""
    base_name, extension = os.path.splitext(filename)
    timestamp = int(time.time() * 1000)
    return f"{base_name}_{timestamp}{extension}"

def load_chat_history(user_id: str, topic_id: str) -> List[Dict]:
    """Load chat history for a specific user and topic"""
    try:
        response = requests.get(f"{API_URL}/chat/history/{user_id}/{topic_id}")
        if response.status_code == 200:
            history = response.json()["history"]
            return history if history else []
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
    return []

def refresh_topics():
    """Refresh the list of available topics"""
    try:
        response = requests.get(f"{API_URL}/topics")
        if response.status_code == 200:
            st.session_state.topics = set(response.json()["topics"])
            return True
    except Exception as e:
        st.error(f"Error loading topics: {str(e)}")
    return False

# Page configuration
st.set_page_config(layout="wide", page_title="Document Chat System")

# Main layout with two columns
column1, column2 = st.columns([1, 2])

# Left column - Document Management
with column1:
    st.header("Document Management")
    
    # Display user ID
    st.info(f"User ID: {st.session_state.user_id}")
    
    # Topic management
    st.subheader("Topics")
    
    # Create new topic
    new_topic = st.text_input("Create New Topic")
    if st.button("Create Topic") and new_topic:
        if new_topic not in st.session_state.topics:
            # Add new topic to session state without querying backend
            st.session_state.topics.add(new_topic)
            st.session_state.current_topic = new_topic
            st.success(f"Created topic: {new_topic}")
            st.rerun()
        else:
            st.warning(f"Topic '{new_topic}' already exists")
    
    # Select existing topic
    if st.session_state.topics:
        topics_list = sorted(list(st.session_state.topics))
        current_index = topics_list.index(st.session_state.current_topic) if st.session_state.current_topic in topics_list else 0
        
        selected_topic = st.selectbox(
            "Select Topic",
            topics_list,
            index=current_index
        )
        
        if selected_topic != st.session_state.current_topic:
            st.session_state.current_topic = selected_topic
            st.rerun()
    else:
        st.info("No topics available. Create a new topic to get started.")
    
    # Document upload section
    if st.session_state.current_topic:
        st.subheader(f"Upload to {st.session_state.current_topic}")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "docx"],
            key=f"uploader_{st.session_state.current_topic}"
        )
        
        if uploaded_file and st.button("Upload Document"):
            with st.spinner("Processing document..."):
                doc_id = generate_unique_doc_id(uploaded_file.name)
                files = {"file": uploaded_file}
                try:
                    response = requests.post(
                        f"{API_URL}/documents/upload",
                        files=files,
                        params={
                            "topic_id": st.session_state.current_topic,
                            "doc_id": doc_id
                        }
                    )
                    if response.status_code == 200:
                        st.success(f"Successfully uploaded: {uploaded_file.name}")
                        st.rerun()
                    else:
                        st.error("Upload failed")
                except Exception as e:
                    st.error(f"Error during upload: {str(e)}")
    
    # Document list
    st.subheader("Documents by Topic")
    for topic in st.session_state.topics:
        with st.expander(f"Topic: {topic}", expanded=(topic == st.session_state.current_topic)):
            try:
                response = requests.get(f"{API_URL}/documents/{topic}")
                if response.status_code == 200:
                    documents = response.json()["documents"]
                    if not documents:
                        st.info("No documents in this topic")
                    else:
                        for doc in documents:
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                col1.write(doc["name"])
                                delete_key = f"del_btn_{topic}_{doc['id']}"
                                if col2.button("Delete", key=delete_key):
                                    try:
                                        del_response = requests.delete(
                                            f"{API_URL}/documents/{topic}/{doc['id']}"
                                        )
                                        if del_response.status_code == 200:
                                            st.success("Document deleted")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting document: {str(e)}")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")

# Right column - Chat Interface
with column2:
    st.header("Chat Interface")
    
    if st.session_state.current_topic:
        st.subheader(f"Chatting in: {st.session_state.current_topic}")

        # Initialize/load chat history for current topic if not already loaded
        if st.session_state.current_topic not in st.session_state.chat_messages:
            st.session_state.chat_messages[st.session_state.current_topic] = load_chat_history(
                st.session_state.user_id,
                st.session_state.current_topic
            )
        
        # Create a container for chat messages
        chat_container = st.container()
        
        # Display current topic's chat history
        with chat_container:
            messages = st.session_state.chat_messages[st.session_state.current_topic]
            for message in messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        st.markdown("---")
        chat_input = st.chat_input("Type your message here...")
        
        if chat_input:
            # Add user message to chat history and display
            user_message = {"role": "human", "content": chat_input}
            st.session_state.chat_messages[st.session_state.current_topic].append(user_message)
            
            # Display user message
            with chat_container:
                with st.chat_message("human"):
                    st.markdown(chat_input)
            
            # Get AI response
            try:
                with st.spinner("AI is thinking..."):
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": chat_input,
                            "user_id": st.session_state.user_id,
                            "topic_id": st.session_state.current_topic
                        },
                        stream=True
                    )
                    
                    if response.status_code == 200:
                        # Create a placeholder for the AI response
                        with chat_container:
                            with st.chat_message("assistant"):
                                message_placeholder = st.empty()
                        
                        # Process the streaming response
                        full_response = ""
                        for line in response.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])  # Skip "data: " prefix
                                        full_response = data["content"]
                                        message_placeholder.markdown(full_response)
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Add AI response to chat history
                        if full_response:
                            ai_message = {"role": "assistant", "content": full_response}
                            st.session_state.chat_messages[st.session_state.current_topic].append(ai_message)
                    else:
                        st.error("Failed to get AI response")
            except Exception as e:
                st.error(f"Error during chat: {str(e)}")
    else:
        st.info("Please select or create a topic to start chatting")