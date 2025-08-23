import streamlit as st
import re
import numpy as np
import pickle
import faiss
from typing import List
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import json
from huggingface_hub import login

# This is the most important function. It loads all heavy components and caches them.
@st.cache_resource
def load_resources():
    """Loads all necessary models, tokenizers, and data indexes."""
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS and BM25 indexes (ensure these files are uploaded to Colab)
    with open("bm25_index_small.pkl", "rb") as f:
        bm25_index_small = pickle.load(f)
    faiss_index_small = faiss.read_index("faiss_index_small.bin")

    # Access the Hugging Face token from the secrets file
    huggingface = st.secrets["huggingface"]
    NGROK_AUTH_TOKEN = st.secrets["NGROK_AUTH_TOKEN"]
    login(token=huggingface)

    # Load RAG Model from Hugging Face Hub
    model_name_rag = "mistralai/Mistral-7B-Instruct-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer_rag = AutoTokenizer.from_pretrained(model_name_rag)
    
    # FIX: Removed the 'device_map="auto"' argument
    model_rag = AutoModelForCausalLM.from_pretrained(
        model_name_rag,
        quantization_config=bnb_config
    )
    
    rag_pipeline = pipeline("text-generation", model=model_rag, tokenizer=tokenizer_rag, max_new_tokens=256, temperature=0.1)

    # For simplicity, we'll use the base RAG model for the RAFT path in this deployment.
    model_ft = model_rag
    tokenizer_ft = tokenizer_rag

    # Load chunk data (ensure rag_chunks_data.json is uploaded)
    with open("rag_chunks_data.json", "r") as f:
        rag_data = json.load(f)
    rag_chunks_small = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in rag_data['small_chunks']]

    index_sets = {"small": {"faiss": faiss_index_small, "bm25": bm25_index_small, "chunks": rag_chunks_small}}

    # Return all the loaded resources
    return embedding_model, index_sets, rag_pipeline, model_ft, tokenizer_ft


# ==============================================================================
# HELPER FUNCTIONS BELOW THIS LINE
# ==============================================================================

##########################
# Step 1: Extend patterns #
##########################

RAFT_PATTERNS = {
    # --- Income Statement ---
    "revenue": [
        r"total income.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "profit after tax": [
        r"profit after tax.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore",
        r"net profit.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "earnings per share": [
        r"earnings per.*?share.*?(`|₹)\s*([\d\.]+)"
    ],

    # --- Balance Sheet ---
    "total assets": [
        r"total assets.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "total liabilities": [
        r"total liabilities.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "deposits": [
        r"total deposits.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "loans": [
        r"advances.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],
    "total borrowings": [
        r"borrowings.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],

    # --- Dividends ---
    "dividend per share": [
        r"dividend of ` ([\d\.]+) per equity share",
        r"dividend of ₹ ([\d\.]+) per equity share",
        r"dividend of ([\d\.]+) per equity share",
        r"dividend.*?([\d\.]+) per share"
    ],
    "total dividend": [
        r"dividend paid by the Bank.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore",
        r"aggregating to.*?(`|₹)\s*([\d,]+\.\d{2})\s*crore"
    ],

    # --- Ratios & Metrics (usually percentages) ---
    "capital adequacy ratio": [
        r"capital adequacy ratio.*?([\d\.]+)%"
    ],
    "gross npa": [
        r"gross npa.*?([\d\.]+)%"
    ],
    "net npa": [
        r"net npa.*?([\d\.]+)%"
    ],
    "return on assets": [
        r"return on assets.*?([\d\.]+)%"
    ],
    "return on equity": [
        r"return on equity.*?([\d\.]+)%"
    ]
}

# Example alias mapping - define as per your use case
metric_aliases = {
    # --- Income Statement Items ---
    "revenue": ["revenue", "total income", "top line", "turnover"],
    "profit after tax": ["profit after tax", "pat", "net profit", "net earnings", "profit for the period"],
    "earnings per share": ["earnings per share", "eps"],

    # --- Balance Sheet Items ---
    "total assets": ["total assets"],
    "total liabilities": ["total liabilities"],
    "deposits": ["deposits", "total deposits"],
    "loans": ["advances", "loans"],
    "total borrowings": ["borrowings", "total borrowings"],

    # --- Dividend-Related ---
    "dividend per share": ["dividend per share"],
    "total dividend": ["dividend paid", "dividend aggregating", "dividend"],

    # --- Key Banking Ratios & Metrics ---
    "capital adequacy ratio": ["capital adequacy ratio", "car"],
    "gross npa": ["gross npa", "gross non-performing assets"],
    "net npa": ["net npa", "net non-performing assets"],
    "return on assets": ["return on assets", "roa"],
    "return on equity": ["return on equity", "roe"]
}

####################################
# Utility: Select top relevant paragraphs
####################################

def select_top_paragraphs(query: str, paragraphs: List[str], top_n: int = 3) -> List[str]:
    query_terms = set(query.lower().split())
    scored = []
    for para in paragraphs:
        para_terms = set(para.lower().split())
        score = len(query_terms & para_terms)
        scored.append((score, para))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))  # Prioritize match count then length
    selected = [para for _, para in scored[:top_n]]
    return selected

#############################
# Convert to number function #
#############################

def convert_to_number(value_str, unit=None):
    """
    Convert a string + unit (lakh, crore, million, etc.) into a float in base units.
    Example: ("1,099.42", "crore") -> 10994200000.0
    """
    num = value_str.replace(",", "")
    try:
        base_val = float(num)
    except ValueError:
        return None

    if unit:
        unit = unit.lower()
        if unit in ["lakh", "lakhs"]:
            base_val *= 1e5
        elif unit in ["crore", "cr", "crores"]:
            base_val *= 1e7
        elif unit == "million":
            base_val *= 1e6
        elif unit == "billion":
            base_val *= 1e9
        elif unit == "trillion":
            base_val *= 1e12
    return base_val


############################################
# Pre-process Query
############################################

def preprocess_query(query: str) -> str:
    """
    Helper function to clean and normalize a query.
    """
    return query.lower().strip()

############################################
# Post-process descriptive answers (truncation or summarization)
############################################

def postprocess_descriptive_answer(answer, max_sentences=3):
    # Simple truncation at sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    if len(sentences) > max_sentences:
        truncated = ' '.join(sentences[:max_sentences]) + " ..."
        return truncated
    return answer


#############################
# clean generated answer
#############################

def clean_generated_answer(text: str) -> str:
    # Find the text that comes after the last instance of the instruction marker
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]

    # Remove any remaining common artifacts and whitespace
    text = text.replace("</s>", "").strip()
    return text

#############################
# Ground Truth Checking
#############################

def is_answer_correct(predicted: str, ground_truth: str, guard_msg: str = None) -> bool:
    """
    Returns True if the predicted answer matches ground_truth.
    Uses a more robust regex to handle numbers and avoid errors.
    """
    # FIX: A more robust regex that ensures a number starts with a digit.
    number_pattern = r"\d[\d,]*\.?\d*"

    # Extract numbers from both the predicted answer and the ground truth
    pred_nums = re.findall(number_pattern, predicted)
    truth_nums = re.findall(number_pattern, ground_truth)

    # Clean the numbers by removing commas before converting to float
    pred_nums_set = {float(n.replace(",", "")) for n in pred_nums}
    truth_nums_set = {float(n.replace(",", "")) for n in truth_nums}

    if not truth_nums_set: # If ground truth has no numbers, fall back to simple string comparison
        return predicted.strip().lower() == ground_truth.strip().lower()

    # Check if any of the key numbers from the ground truth are present in the prediction
    return bool(pred_nums_set & truth_nums_set)


# ==============================================================================
# Main FUNCTIONS BELOW THIS LINE
# ==============================================================================


#############################
# hybrid Retrieval
#############################
def hybrid_retrieval(query, index_sets, embedding_model, strategy_name="small", top_k=5):
    """
    Performs hybrid retrieval using the specified chunking strategy ('small' or 'large').
    It combines dense (FAISS) and sparse (BM25) search results using Reciprocal Rank Fusion (RRF).
    """
    # 1. Select the correct set of indexes and chunks based on the chosen strategy
    if strategy_name not in index_sets:
        print(f" Warning: Strategy '{strategy_name}' not found. Defaulting to 'small'.")
        strategy_name = "small"

    strategy = index_sets[strategy_name]
    faiss_index = strategy.get("faiss")
    bm25_index = strategy.get("bm25")
    chunks = strategy.get("chunks")

    if faiss_index is None or bm25_index is None:
        print("Indexes are not available. Skipping retrieval.")
        return [], [], []

    # 2. Preprocess query
    cleaned_query = preprocess_query(query)

    # 3. Dense retrieval (FAISS)
    query_emb = embedding_model.encode([cleaned_query])[0]
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    distances, dense_indices = faiss_index.search(np.array([query_emb_norm]).astype('float32'), top_k)
    dense_results = [{'doc_index': i, 'score': s} for i, s in zip(dense_indices[0], distances[0]) if i != -1]

    # 4. Sparse retrieval (BM25)
    tokenized_query = cleaned_query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    sparse_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    sparse_results = [{'doc_index': i, 'score': bm25_scores[i]} for i in sparse_top_indices]

    # 5. Reciprocal Rank Fusion (RRF)
    fused_scores = {}
    k_rrf = 60  # Tunable parameter

    for rank, res in enumerate(dense_results):
        idx = res['doc_index']
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

    for rank, res in enumerate(sparse_results):
        idx = res['doc_index']
        fused_scores[idx] = fused_scores.get(idx, 0) + 1 / (k_rrf + rank + 1)

    # 6. Sort documents by fused score
    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    top_indices = sorted_indices[:top_k]

    # 7. Retrieve documents, add fused_score in metadata
    final_docs = []
    seen_indices = set()
    for idx in top_indices:
        if idx not in seen_indices:
            doc = chunks[idx]
            doc.metadata['fused_score'] = fused_scores[idx]
            final_docs.append(doc)
            seen_indices.add(idx)

    retrieved_texts = [doc.page_content for doc in final_docs]

    return final_docs, retrieved_texts, None

#############################
# Normalize Years
#############################
def normalize_years_in_query(query: str) -> list[str]:
    """
    Finds years in various formats (e.g., 2024, FY25, 23-24) in a query
    and returns them as a sorted list of unique four-digit strings.
    """
    found_years = set()

    # Pattern 1: Find four-digit years like 2023, 2024
    for match in re.findall(r"\b(20\d{2})\b", query):
        found_years.add(match)

    # Pattern 2: Find formats like FY24, FY 25, Financial Year 24
    for match in re.findall(r"\b(?:FY|Financial Year)\s*(\d{2,4})\b", query, re.IGNORECASE):
        year = f"20{match}" if len(match) == 2 else match
        found_years.add(year)

    # Pattern 3: Find year ranges like 23-24 or 2023-24
    for match in re.findall(r"\b(\d{2,4})[-–](\d{2,4})\b", query):
        start, end = match
        start_full = f"20{start}" if len(start) == 2 else start
        end_full = f"20{end}" if len(end) == 2 else end
        found_years.add(start_full)
        found_years.add(end_full)

    return sorted(list(found_years))

#############################
# Extract comparative values
#############################
def extract_comparative_values(context, metric_aliases):
    """
    Finds sentences with a "value A compared to value B" structure.
    This robust version finds all numbers and years on a line and pairs them.
    Returns a tuple: (later_year_value, earlier_year_value)
    """
    # Helper to convert string like "` 500 crore" to a numeric value
    def to_float(s):
        try:
            num_part = re.search(r'[\d,\.]+', s).group()
            num = float(num_part.replace(",", ""))
            if "crore" in s.lower():
                num *= 10_000_000
            elif "lakh" in s.lower():
                num *= 100_000
            return num
        except (AttributeError, ValueError):
            return None

    for line in context.split('\n'):
        # 1. Check for prerequisite keywords to ensure the line is relevant
        if not any(alias in line.lower() for alias in metric_aliases):
            continue
        if not any(comp in line.lower() for comp in ["as compared to", "from"]):
            continue

        # 2. Find all number-like and year-like strings on the line
        num_pattern = r"[`₹]?\s*[\d,]+\.\d{2}(?:\s*crore|\s*lakh)?"
        year_pattern = r"\b20\d{2}\b"

        numbers_found = re.findall(num_pattern, line)
        years_found = re.findall(year_pattern, line)

        # 3. If we find exactly two of each, we have a very strong candidate
        if len(numbers_found) == 2 and len(years_found) == 2:
            try:
                val1 = to_float(numbers_found[0])
                val2 = to_float(numbers_found[1])
                year1 = int(years_found[0])
                year2 = int(years_found[1])

                if val1 is None or val2 is None:
                    continue

                # 4. Associate and return in the correct order (later year, earlier year)
                if year1 > year2:
                    return (val1, val2)
                else:
                    return (val2, val1)
            except (ValueError, TypeError, IndexError):
                # If anything goes wrong with this line, just move to the next
                continue

    # If no suitable line was found in the entire context, return None
    return None

#############################
# RAG Generative
#############################
def generate_rag_response(query, retrieved_texts, rag_pipeline, tokenizer_rag):
    context = "\n".join(retrieved_texts)
    messages = [
        {"role": "system", "content": "You are a helpful financial assistant. Answer ONLY based on the provided context. Be concise. If the answer is not in the context, say 'Information not available.'"},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nDirect Answer:"}
    ]
    prompt = tokenizer_rag.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = rag_pipeline(prompt)
    return response[0]['generated_text'].strip()

#############################
# RAFT Generative
#############################
def generate_raft_response(query, retrieved_texts, model_ft, tokenizer_ft, extract_comparative_values_func, metric_aliases, years_found):
    """
    Generates a response using the RAFT pipeline.
    This corrected version uses the arguments passed into it, avoiding redundant work.
    """
    # 1. Use the retrieved_texts that are passed in directly.
    context_str = "\n".join(retrieved_texts)

    # 2. Perform direct extraction using the provided context and helper functions.
    query_lower = query.lower()
    metric_key = next((k for k, aliases in metric_aliases.items() if any(alias in query_lower for alias in aliases)), None)
    is_change_query = any(w in query_lower for w in ["change", "difference", "compare", "between", "year-on-year"])

    # Attempt direct extraction for comparative questions.
    if metric_key and len(years_found) == 2 and is_change_query:
        # Use the function that was passed in as an argument.
        values = extract_comparative_values_func(context_str, metric_aliases[metric_key])
        if values:
            later_val, earlier_val = values
            change = later_val - earlier_val
            # Return the calculated answer and high confidence.
            return f"₹{change:,.2f}", context_str, None, 0.98

    # 3. If direct extraction fails, fall back to the LLM using the provided model and tokenizer.
    messages = [
        {"role": "system", "content": "You are a financial analyst. Answer using ONLY the provided context. If a calculation is needed, perform it and provide only the final numerical result."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nDirect Numerical Answer:"}
    ]
    prompt = tokenizer_ft.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_ft(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model_ft.device)

    try:
        with torch.no_grad():
            outputs = model_ft.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer_ft.eos_token_id)

        generated_text = tokenizer_ft.decode(outputs[0], skip_special_tokens=True).strip()
        # Assumes a 'clean_generated_answer' function is available.
        final_answer = clean_generated_answer(generated_text)
        return final_answer, context_str, prompt, 0.7
    except Exception as e:
        print(f"Error during RAFT generation: {e}")
        return "Information not available", context_str, None, 0.0

#############################
# Post Processing the generated text
#############################
def postprocess_generated_text(raw_text, prompt_marker, question=None, years_found=None):
    if not raw_text:
        return ""

    # --- Clean model output ---
    # Only split if a valid prompt_marker is provided and exists in the text
    if prompt_marker and prompt_marker in raw_text:
        parts = raw_text.split(prompt_marker)
        raw_text = parts[-1]

    tokens_to_remove = ["[INST]", "[/INST]", "INST", "</s>", "<s>"]
    for token in tokens_to_remove:
        raw_text = raw_text.replace(token, "")

    raw_text = " ".join(raw_text.split()).strip()

    # If the text is just a number, it's likely from direct extraction, so return as is.
    if re.match(r'^[\d,.]+$', raw_text):
        return raw_text

    # (The rest of the function remains the same)
    words = raw_text.split()
    while words and len(words[-1]) < 2:
        words.pop()
    raw_text = " ".join(words)

    if len(raw_text) < 3:
        return "Information not available"

    # --------- STEP 2: Financial formatting ---------
    formatted = raw_text
    if question:
        q = question.lower()
        num_match = re.search(r"([\d,]+(?:\.\d+)?)\s*(crore|lakh|million|billion|trillion)", formatted, re.IGNORECASE)
        if num_match:
            value, unit = num_match.groups()
            formatted = f"₹{value} {unit}"

        # Dynamic year handling
        start_year, end_year = None, None
        if years_found and isinstance(years_found, list):
            years_found = sorted(list(set(years_found))) # Ensure unique and sorted
            if len(years_found) == 1:
                start_year = years_found[0]
            elif len(years_found) >= 2:
                start_year, end_year = years_found[0], years_found[1]

        # Auto-label based on question type
        if "revenue" in q:
            if end_year:
                formatted = f"Change in Total Revenue from FY {start_year} to FY {end_year} was {formatted}."
            elif start_year:
                formatted = f"Total Revenue for FY {start_year} was {formatted}."
        elif "profit" in q:
            if end_year:
                formatted = f"Change in Net Profit after Tax from FY {start_year} to FY {end_year} was {formatted}."
            elif start_year:
                formatted = f"Net Profit after Tax for FY {start_year} was {formatted}."
        elif "asset" in q:
            if end_year:
                formatted = f"Change in Total Assets from FY {start_year} to FY {end_year} was {formatted}."
            elif start_year:
                formatted = f"Total Assets for FY {start_year} were {formatted}."
        elif "liabilit" in q:
            if end_year:
                formatted = f"Change in Total Liabilities from FY {start_year} to FY {end_year} was {formatted}."
            elif start_year:
                formatted = f"Total Liabilities for FY {start_year} were {formatted}."

    return formatted
