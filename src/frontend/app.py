import streamlit as st
import requests
import json
import asyncio

API_URL = "http://localhost:8000"

st.title("Document Intelligence System")

# Initialize session state for documents
if 'documents' not in st.session_state:
    st.session_state.documents = {}

# File upload section
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "png"])
if uploaded_file is not None:
    if st.button("Upload and Index"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_URL}/upload", files=files)
        doc_id = response.json()["doc_id"]
        st.session_state.documents[doc_id] = uploaded_file.name
        st.success(f"File uploaded and indexed successfully. Document ID: {doc_id}")

# Document management section
st.subheader("Managed Documents")
for doc_id, doc_name in st.session_state.documents.items():
    col1, col2 = st.columns([3, 1])
    col1.write(f"{doc_id}: {doc_name}")
    if col2.button("Remove", key=f"remove_{doc_id}"):
        response = requests.delete(f"{API_URL}/document/{doc_id}")
        if response.status_code == 200:
            del st.session_state.documents[doc_id]
            st.success(f"Document {doc_id} removed successfully")
            st.experimental_rerun()

# Only show other functionalities if documents have been uploaded
if st.session_state.documents:
    selected_doc_id = st.selectbox("Select a document", list(st.session_state.documents.keys()))

    st.subheader("Search")
    search_query = st.text_input("Enter your search query")
    if st.button("Search"):
        response = requests.post(f"{API_URL}/search", json={"text": search_query})
        results = response.json()
        st.write(results)

    st.subheader("Question Answering")
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        response = requests.post(f"{API_URL}/answer", json={"text": question}, params={"doc_id": selected_doc_id})
        answer = response.json()
        st.write(answer)

    if st.button("Get Summary"):
        response = requests.get(f"{API_URL}/summary/{selected_doc_id}")
        summary = response.json()["summary"]
        st.write(summary)

    if st.button("Detect Bias"):
        response = requests.get(f"{API_URL}/bias/{selected_doc_id}")
        bias = response.json()
        st.write(bias)

    st.subheader("Fine-tune Model")
    num_qa_pairs = st.number_input("Number of question-answer pairs", min_value=1, max_value=10, value=3)
    qa_pairs = []
    for i in range(num_qa_pairs):
        question = st.text_input(f"Question {i+1}")
        answer = st.text_input(f"Answer {i+1}")
        qa_pairs.append({"question": question, "answer": answer})

    if st.button("Fine-tune Model"):
        response = requests.post(f"{API_URL}/fine-tune/{selected_doc_id}", json=qa_pairs)
        st.write(response.json()["message"])