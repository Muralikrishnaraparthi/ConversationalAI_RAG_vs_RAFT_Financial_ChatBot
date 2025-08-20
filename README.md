Group 99 Financial QA Chatbot
Overview
This project implements a Financial Question Answering (QA) Chatbot for HDFC Bank's financial reports using advanced Conversational AI techniques. It combines Retrieval-Augmented Generation (RAG) and Retrieval-Augmented Fine-Tuning (RAFT) to provide accurate and context-aware answers to financial queries.

Project Members
Name	ID
Aparnaram Kasireddy	2023AC05145
K Niranjan Babu	2023AC05464
Lakshmi Mrudula Maddi	2023AC05138
Muralikrishna Raparthi	2023AC05208
Rajamohan Naidu	2023AC05064

Architecture Diagram:
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/02650762-48d4-4acd-9199-51b61fdc3bdb" />


Features
Accurate extraction of financial data from HDFC Bank’s reports.

Hybrid retrieval using dense (FAISS) and sparse (BM25) methods combined via Reciprocal Rank Fusion.

Dual chunking strategy for optimized context retrieval:

Small chunks (512 tokens) for precise retrieval.

Large chunks (2048 tokens) for rich generative context.

RAFT fine-tuning to reduce hallucinations and improve factual answer generation.

Interactive user interfaces available in CLI and Streamlit web app.

Integrated guardrails to detect and limit hallucinations.

Technologies & Techniques
Hybrid Retrieval: Combines dense vector search (FAISS) and keyword matching (BM25).

RAG Model: Base Mistral-7B model fine-tuned for on-the-fly contextual response generation.

RAFT Model: Fine-tuned Mistral-7B model with LoRA adapters, trained offline using augmented financial Q&A pairs.

Regex and Rule-Based Extraction: Used for precise number and metric extraction from retrieved texts.

Streamlit UI: Lightweight web interface for querying the chatbot.

Python Libraries: Transformers, SentenceTransformers, FAISS, rank_bm25, PyTorch, Streamlit.

Architecture
Data Preparation & PDF Processing

Text extraction and cleaning from HDFC Bank financial PDFs.

Chunking into both small and large text segments.

Hybrid Indexing

Creation of sparse (BM25) and dense (FAISS) indices for efficient retrieval.

Model Pipelines

RAG for immediate generation from retrieved context.

RAFT for fine-tuned generation with reduced hallucination.

Hybrid Retrieval

Fuse results from both retrieval methods to find the most relevant contexts.

Interactive UI

Query input via CLI or Streamlit.

Option to choose between RAG and RAFT models.

Display confidence, response time, and context snippet.

How To Use
Streamlit Web App
Run the Streamlit app with all prerequisites.

Enter questions related to HDFC Bank financial data.

Choose between RAFT (Fine-Tuned) or RAG (Base Model).

Get answers instantly with relevant context shown on demand.

CLI Interface
Launch the CLI script.

Input financial queries.

Select model mode (RAG/RAFT).

Receive answers with confidence scores and retrieval times.

Sample Questions Tested
What was the revenue from operations for the year ended March 31, 2024?

What was the change in Profit After Tax from FY 2024 to FY 2025?

How did the cost-to-income ratio of the bank change over the last two years?

What was the dividend paid for the financial year 2024?

What was the dividend per share for the financial year 2025?

Key Performance Insights
The hybrid retrieval approach achieves improved relevance by leveraging both keyword and semantic search.

RAFT reduces hallucinations with guardrails and improves accuracy on numeric financial data.

Streamlit and CLI interfaces provide flexible user interaction modes.

Real-time confidence scoring and context show help validate answers.

Installation & Setup
Clone the repository.

Install dependencies from requirements.txt or manual installation:

streamlit, sentence-transformers, faiss, rank_bm25, transformers, torch, etc.

Place the preprocessed data files (faiss_index_small.bin, bm25_index_small.pkl, rag_chunks_data.json) in the working directory.

Run the Streamlit app:

text
streamlit run app.py
Or run the CLI interface by executing the CLI script.

Streamlit UI:
<img width="1500" height="843" alt="image" src="https://github.com/user-attachments/assets/673645e6-fdc6-4654-b47b-9c1e0175c34b" />
