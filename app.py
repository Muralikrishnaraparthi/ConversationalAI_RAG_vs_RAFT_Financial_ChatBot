import streamlit as st
import time
import numpy as np
from utils import (
    load_resources, 
    hybrid_retrieval, 
    normalize_years_in_query,
    extract_comparative_values,
    generate_rag_response,
    postprocess_generated_text,
    metric_aliases
)

# --- App Configuration ---
st.set_page_config(
    page_title="Group 99 | Financial QA", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Access the Hugging Face token from the secrets file
huggingface = st.secrets["huggingface"]
NGROK_AUTH_TOKEN = st.secrets["NGROK_AUTH_TOKEN"]

# --- Load Models and Data ---
with st.spinner("Loading financial models and data... This may take a moment on first startup."):
    embedding_model, index_sets, rag_pipeline, model_ft, tokenizer_ft = load_resources()

# --- UI Components ---
st.title("🤖 Group 99: Financial QA Chatbot")
st.sidebar.header("Query Options")

# FIX: Removed the unsupported 'caption' argument from the line below
mode_selection = st.sidebar.radio(
    "Choose a Model:", 
    ("RAFT (Fine-Tuned)", "RAG (Base Model)")
)
mode = "RAFT" if "RAFT" in mode_selection else "RAG"

st.sidebar.markdown("---") 
with st.sidebar.expander("Project Group Members"):
    st.markdown("""
    | Name                      | ID            |
    | :------------------------ | :------------ |
    | APARNARAM KASIREDDY       | 2023AC05145   |
    | K NIRANJAN BABU           | 2023AC05464   |
    | LAKSHMI MRUDULA MADDI     | 2023AC05138   |
    | MURALIKRISHNA RAPARTHI    | 2023AC05208   |
    | RAJAMOHAN NAIDU           | 2023AC05064   |
    """)

query = st.text_input("Ask a question about HDFC Bank's financial reports:", "")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Finding relevant documents and generating an answer..."):
            start_time = time.time()
            
            retrieved_docs, retrieved_texts, _ = hybrid_retrieval(query, index_sets, embedding_model, top_k=5)
            context_used = "\n".join(retrieved_texts)
            
            if not retrieved_texts:
                st.error("Could not find relevant context to answer this question.")
            else:
                years_found = normalize_years_in_query(query)
                raw_answer = None
                model_confidence = 0.0

                query_lower = query.lower()
                metric_key = next((k for k, aliases in metric_aliases.items() if any(alias in query_lower for alias in aliases)), None)
                is_change_query = any(w in query_lower for w in ["change", "difference", "compare", "between", "year-on-year"])

                if metric_key and len(years_found) >= 2 and is_change_query:
                    values = extract_comparative_values(context_used, metric_aliases[metric_key])
                    if values:
                        later_val, earlier_val = values
                        change = later_val - earlier_val
                        raw_answer = f"₹{change:,.2f}"
                        model_confidence = 0.98

                if raw_answer is None:
                    raw_answer = generate_rag_response(query, retrieved_texts, rag_pipeline, tokenizer_ft)
                    scores = [doc.metadata.get('fused_score', 0) for doc in retrieved_docs]
                    model_confidence = np.mean(scores) if scores else 0.0

                final_answer = postprocess_generated_text(raw_answer, "[/INST]", query, years_found=years_found)
                end_time = time.time()

                st.success("Answer")
                st.markdown(f"### {final_answer}")
                
                col1, col2 = st.columns(2)
                col1.metric(label="Response Time", value=f"{end_time - start_time:.2f}s")
                col2.metric(label="Confidence Score", value=f"{model_confidence:.2%}")

                with st.expander("Show Retrieved Context"):
                    st.text(context_used)
