import streamlit as st
import tempfile
import os
from core.document_store import get_document_store
from core.ingest import ingest_pdf
from services.outcome_service import run_outcome
from services.bias_service import run_bias
from services.chat_service import run_chat

st.set_page_config(page_title="Proposal Evaluator", layout="wide")

st.title("AI Proposal Evaluation System")

def get_namespace(filename: str) -> str:
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext.replace(" ", "_")


uploaded = st.file_uploader("Upload Proposal PDF", type=["pdf"])

if uploaded:

    namespace = get_namespace(uploaded.name)

    st.info(f"Using namespace: {namespace}")

   
    document_store = get_document_store(namespace=namespace)

  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        temp_path = f.name

    ingest_pdf(temp_path, document_store)

    st.success("PDF indexed successfully")
    st.write(f"Total vectors in this namespace: {document_store.count_documents()}")

  
    st.session_state.namespace = namespace
    st.session_state.document_store = document_store

if "document_store" in st.session_state:

    document_store = st.session_state.document_store

    tab1, tab2, tab3 = st.tabs(["Outcome Predictor", "Bias Predictor", "Chat"])

    with tab1:
        notes = st.text_area("Additional Notes")
        if st.button("Evaluate Outcome"):
            result = run_outcome(document_store, notes)
            st.json(result)

    with tab2:
        bias = st.text_area("Department Bias")
        if st.button("Analyze Bias"):
            result = run_bias(document_store, bias)
            st.json(result)

    with tab3:
        if "chat" not in st.session_state:
            st.session_state.chat = ""

        query = st.text_input("Ask something")

        if st.button("Send"):
            reply = run_chat(document_store, st.session_state.chat, query)
            st.session_state.chat += f"\nUser: {query}\nAI: {reply}"

        st.text_area("Conversation", st.session_state.chat, height=300)