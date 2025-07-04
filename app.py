import os
import streamlit as st
from rag import DocumentProcessor, RAGAdvisor


st.set_page_config(page_title="RAG GenAI Advisor", layout="wide")
st.title("RAG GenAI Advisor")

model_name = st.sidebar.text_input("Model Name", value="mistralai/Mistral-7B-Instruct-v0.2")
hf_token = st.sidebar.text_input("Hugging Face Token (optional)", type="password")
role = st.sidebar.text_input("Advisor Role", value="General Advisor")

uploaded_file = st.file_uploader("Upload PDF or PPTX", type=["pdf", "ppt", "pptx"])

if 'processor' not in st.session_state:
    st.session_state['processor'] = None
if 'advisor' not in st.session_state:
    st.session_state['advisor'] = None
if 'history' not in st.session_state:
    st.session_state['history'] = []

if uploaded_file:
    file_path = os.path.join(".data", uploaded_file.name)
    os.makedirs(".data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state['processor'] = DocumentProcessor()
    st.session_state['processor'].load_document(file_path)
    st.success("Document processed and indexed")

if st.session_state.get('processor'):
    if st.session_state.get('advisor') is None or st.session_state['advisor'].role != role:
        st.session_state['advisor'] = RAGAdvisor(st.session_state['processor'], model_name, role, hf_token)

    instruction = st.text_input("Enter instruction for the advisor")
    if st.button("Submit") and instruction:
        response = st.session_state['advisor'].run(instruction)
        st.session_state['history'].append((instruction, response))

    st.subheader("Conversation")
    for i, (question, answer) in enumerate(st.session_state['history']):
        st.markdown(f"**Q{i+1}: {question}**")
        st.markdown(f"**A{i+1}: {answer}**")

else:
    st.info("Please upload a document to begin")
