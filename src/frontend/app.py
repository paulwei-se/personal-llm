import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.title("Document Intelligence System")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "png"])
if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    doc_id = response.json()["doc_id"]
    st.success(f"File uploaded successfully. Document ID: {doc_id}")

    st.subheader("Search")
    search_query = st.text_input("Enter your search query")
    if st.button("Search"):
        response = requests.post(f"{API_URL}/search", json={"text": search_query})
        results = response.json()
        st.write(results)

    st.subheader("Question Answering")
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        response = requests.post(f"{API_URL}/answer", json={"text": question}, params={"doc_id": doc_id})
        answer = response.json()
        st.write(answer)

    if st.button("Get Summary"):
        response = requests.get(f"{API_URL}/summary/{doc_id}")
        summary = response.json()["summary"]
        st.write(summary)

    if st.button("Detect Bias"):
        response = requests.get(f"{API_URL}/bias/{doc_id}")
        bias = response.json()
        st.write(bias)
