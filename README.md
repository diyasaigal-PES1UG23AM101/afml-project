# Tulu Language Preservation RAG System

This project is an advanced Retrieval-Augmented Generation (RAG) system designed to provide accurate and helpful answers about the Tulu language and culture. It leverages modern AI techniques to address the challenges of working with a low-resource language.

## ✨ Features

*   **Intelligent Query Routing**: The system first classifies a user's query. Simple conversational questions are handled directly by the LLM, while factual questions are routed to the RAG pipeline for grounded answers.
*   **Self-Correcting Fallback**: If the retrieved documents do not contain the answer to a factual question, the system recognizes this failure and falls back to the LLM's general knowledge to provide a helpful response.
*   **Grounded Answers**: For factual questions, answers are generated based on a curated knowledge base of Tulu texts, with sources provided to the user.
*   **Multi-Language Support**: The pipeline is designed to handle queries and generate responses in English, Tulu, or a combination of both.
*   **Optimized for Performance**: Utilizes the fast Groq API for real-time answer generation.

## 🧠 Pipeline Architecture

The system employs a multi-step process to ensure robust and relevant answers:

1.  **Query Classification**: The user's input is first analyzed to determine if it is `conversational` or `factual`.
2.  **Conversational Path**: If conversational, the query is sent directly to the LLM, bypassing the retrieval steps.
3.  **Factual RAG Path**:
    *   **Retrieve**: A set of candidate passages is retrieved from a FAISS vector store using a sentence-transformer model.
    *   **Rerank**: A cross-encoder model reranks the candidates to find the most relevant passages.
    *   **Generate**: The top passages and the user's question are formatted into a prompt and sent to the LLM (Groq Llama 3.1) to generate a grounded answer.
4.  **Self-Correction**: The generated answer is checked. If it is a generic "I don't know" response, the system triggers a fallback.
5.  **Fallback Mechanism**: The original query is sent to the LLM *without* the retrieved context, allowing it to use its general knowledge to provide a more helpful answer.

## 📁 Project Structure
afml-project/

├── app/

│ └── app.py

├── data/

│ └── processed/

│ ├── all_passages.jsonl 

│ └── faiss_index.bin 

├── src/

│ ├── retriever.py 

│ ├── reranker.py 

│ ├── generator.py

│ ├── pipeline.py 

│ ├── prompts.py

│ └── utils.py

├── .env

├── .gitignore

└── requirements.txt

## 🚀 Setup and Installation

### 1. Prerequisites
*   Python 3.8+
*   Git

### 2. Clone the Repository
`git clone https://github.com/diyasaigal-PES1UG23AM101/afml-project.git`

`cd afml-project`

### 3. Set Up a Virtual Environment (Recommended)
# For Windows
`python -m venv venv
venv\Scripts\activate`

# For macOS/Linux
`python -m venv venv
source venv/bin/activate`

# Install Dependencies
`pip install -r requirements.txt`

# 5. Set Up API Keys
* This project requires an API key from Groq.
* Create a file named `.env` in the root of the `afml-project` directory.
* Go to `https://console.groq.com/` to get your free API key.
* Add your key to the `.env` file like this:
`GROQ_API_KEY="gsk_YourSecretGroqApiKeyHere"`

# ▶️ Running the Application
Once the setup is complete, run the Streamlit app from the project's root directory:

`streamlit run app/app.py`

Open your web browser and navigate to the local URL provided (usually http://localhost:8501).

Note: Use `faiss-gpu` instead of `faiss-cpu` if you have a compatible NVIDIA GPU and CUDA installed.
