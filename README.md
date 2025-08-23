# Financial QA CLI - Group 99 Financial QA Chatbot

## Overview

This project implements a Financial Question Answering (QA) Chatbot for HDFC Bank's financial reports using advanced Conversational AI techniques. It combines Retrieval-Augmented Generation (RAG) and Retrieval-Augmented Fine-Tuning (RAFT) to provide accurate and context-aware answers to financial queries.

This command-line interface (CLI) application acts as a financial question-answering system that leverages two different models—a Retrieval-Augmented Generation (RAG) model and a fine-tuned RAFT (Retrieval-Augmented Fine-Tuning) model—to provide accurate answers based on financial documents.

## Project Members

| Name | ID |
|------|-----|
| Aparnaram Kasireddy | 2023AC05145 |
| K Niranjan Babu | 2023AC05464 |
| Lakshmi Mrudula Maddi | 2023AC05138 |
| Muralikrishna Raparthi | 2023AC05208 |
| Rajamohan Naidu | 2023AC05064 |

## Features

The application can:
* Answer user queries on financial data with high accuracy
* Block irrelevant or out-of-scope questions
* Compare the performance of RAG and RAFT models in real-time
* Extract accurate financial data from HDFC Bank's reports
* Implement hybrid retrieval using dense (FAISS) and sparse (BM25) methods combined via Reciprocal Rank Fusion
* Utilize dual chunking strategy for optimized context retrieval:
  - Small chunks (512 tokens) for precise retrieval
  - Large chunks (2048 tokens) for rich generative context
* RAFT fine-tuning to reduce hallucinations and improve factual answer generation
* Interactive user interfaces available in CLI and Streamlit web app
* Integrated guardrails to detect and limit hallucinations

## Technologies & Techniques

* **Hybrid Retrieval**: Combines dense vector search (FAISS) and keyword matching (BM25)
* **RAG Model**: Base Mistral-7B model fine-tuned for on-the-fly contextual response generation
* **RAFT Model**: Fine-tuned Mistral-7B model with LoRA adapters, trained offline using augmented financial Q&A pairs
* **Regex and Rule-Based Extraction**: Used for precise number and metric extraction from retrieved texts
* **Streamlit UI**: Lightweight web interface for querying the chatbot
* **Python Libraries**: Transformers, SentenceTransformers, FAISS, rank_bm25, PyTorch, Streamlit

## Architecture

Architecture Diagram:
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/02650762-48d4-4acd-9199-51b61fdc3bdb" />

The system follows a comprehensive architecture with the following components:

### 1. Data Preparation & PDF Processing
* Text extraction and cleaning from HDFC Bank financial PDFs
* Chunking into both small and large text segments

### 2. Hybrid Indexing
* Creation of sparse (BM25) and dense (FAISS) indices for efficient retrieval

### 3. Model Pipelines
* RAG for immediate generation from retrieved context
* RAFT for fine-tuned generation with reduced hallucination

### 4. Hybrid Retrieval
* Fuse results from both retrieval methods to find the most relevant contexts

### 5. Interactive UI
* Query input via CLI or Streamlit
* Option to choose between RAG and RAFT models
* Display confidence, response time, and context snippet

## Requirements

To run this application, you need to have Python and the necessary libraries installed. All required packages are listed in the `requirements.txt` file.

* **Python**: Version 3.8 or higher is recommended
* **Hugging Face Access Token**: The application uses a gated model from Hugging Face. You must have an account and be logged in to your local machine using the CLI

* **Required Libraries**: streamlit, sentence-transformers, faiss, rank_bm25, transformers, torch, etc.
* **Data Files**: Place the preprocessed data files (`faiss_index_small.bin`, `bm25_index_small.pkl`, `rag_chunks_data.json`) in the working directory

## Setup and Installation

Follow these steps to set up the project on your local machine.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Muralikrishnaraparthi/ConversationalAI_RAG_vs_RAFT_Financial_ChatBot.git
    cd ConversationalAI_RAG_vs_RAFT_Financial_ChatBot
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment**
    * On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Hugging Face Login**
    Log in to your Hugging Face account via the command line to authenticate for gated models.
    ```bash
    huggingface-cli login
    ```

6.  **Data Setup**
    Ensure the preprocessed data files are in your working directory:
    * `faiss_index_small.bin`
    * `bm25_index_small.pkl`
    * `rag_chunks_data.json`

## How to Run the Application

This project can be run in two ways: as a command-line interface (CLI) and as a Streamlit web application.

### Streamlit Web App Mode

**Set Up Your Secrets:**

Create a folder named `.streamlit` in the project's root directory.

Inside this folder, create a file named `secrets.toml`.

Add your tokens to the file using the following format:

```ini
# .streamlit/secrets.toml
huggingface = "hf_your_hugging_face_token_here"
NGROK_AUTH_TOKEN = "your_ngrok_auth_token_here"
```

**Run the App:**
From your terminal, run the Streamlit command.

```bash
streamlit run app.py
```

The app will open in your web browser where you can:
* Enter questions related to HDFC Bank financial data
* Choose between RAFT (Fine-Tuned) or RAG (Base Model)
* Get answers instantly with relevant context shown on demand

### CLI Mode

Run the `Group_99_RAG_vs_FT.ipynb` notebook file completely

The CLI will:
* Prompt you to enter a question and a model choice (RAG or RAFT)
* Input financial queries
* Select model mode (RAG/RAFT)
* Receive answers with confidence scores and retrieval times

Type `exit` to quit the application.

## Sample Questions Tested

Here are some example questions you can ask the system:

* "What was the revenue from operations for the year ended March 31, 2024?"
* "What was the change in Profit After Tax from FY 2024 to FY 2025?"
* "How did the cost-to-income ratio of the bank change over the last two years?"
* "What was the dividend paid for the financial year 2024?"
* "What was the dividend per share for the financial year 2025?"

## Key Performance Insights

* The hybrid retrieval approach achieves improved relevance by leveraging both keyword and semantic search
* RAFT reduces hallucinations with guardrails and improves accuracy on numeric financial data
* Streamlit and CLI interfaces provide flexible user interaction modes
* Real-time confidence scoring and context show help validate answers

## Project Structure

* `app.py`: The main script that handles user interaction and runs the CLI or Streamlit app
* `utils.py`: Contains the core logic, including data loading, model inference, and confidence scoring
* `requirements.txt`: A list of all required Python libraries
* `README.md`: This file
* `.gitignore`: Prevents sensitive files like secrets.toml from being committed to Git
* `Group_99_RAG_vs_FT.ipynb` : Notebook for end to end code
* `faiss_index_small.bin`: Preprocessed FAISS index for dense retrieval
* `bm25_index_small.pkl`: Preprocessed BM25 index for sparse retrieval
* `rag_chunks_data.json`: Chunked data for context retrieval


## Contributing

Contributions are welcome! If you find a bug or have an idea for an improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.