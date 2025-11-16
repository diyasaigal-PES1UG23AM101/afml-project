# Tulu Language RAG System

A Retrieval-Augmented Generation (RAG) system for question-answering in Tulu language, developed as part of the Advanced Foundation Models Lab (AFML) course.

## Overview

This project implements a multilingual RAG system that can answer questions about Tulu language content. Users can ask questions in English or Tulu and receive responses in their preferred language (English, Tulu, or both).

## Features

- **Multilingual Support**: Ask questions in English or Tulu, get answers in your preferred language
- **Semantic Search**: Uses LaBSE embeddings for cross-lingual retrieval
- **Re-ranking**: Improves retrieval quality with a dedicated re-ranking step
- **LLM Generation**: Powered by OpenAI GPT-4o-mini for natural language generation
- **Translation Models**: Support for GPT-4, mBART, and mT5 translation models
- **Fine-tuning**: Fine-tune translation models on domain-specific data
- **Evaluation Metrics**: BLEU and METEOR scores for translation quality
- **Web Interface**: Interactive Streamlit application with advanced configuration
- **Retrieval-Augmented Prompts**: Context-aware answer generation

## Architecture

### RAG Pipeline

1. **Retrieval** (`src/retriever.py`)
   - Encodes queries using LaBSE sentence transformer
   - Searches FAISS index for semantically similar passages
   - Returns top-k candidate passages

2. **Re-ranking** (`src/reranker.py`)
   - Re-scores retrieved passages for better relevance
   - Uses cross-encoder model for accurate ranking
   - Reorders candidates based on query-passage similarity

3. **Generation** (`src/generator.py`)
   - Uses OpenAI GPT-4o-mini to generate contextual answers
   - Supports prompts for English and Tulu responses

4. **Translation** (`src/rag_pipeline.py`)
   - Multi-model support: GPT-4 API, mBART, mT5
   - Fine-tuning capability for domain adaptation
   - Retrieval-augmented translation prompts

5. **Evaluation** (`src/evaluate_translation.py`)
   - BLEU score computation (1-4 grams)
   - METEOR score for semantic similarity
   - Corpus-level and sentence-level metrics

### Data Collection

The `diya/src/` directory contains scripts for building the Tulu corpus:

- **`scrape_wiki.py`**: Scrapes articles from Tulu Wikipedia
- **`scrape_news.py`**: Extracts content from Tulu news websites
- **`add_bible.py`**: Adds Bible content in Tulu
- **`process_raw.py`**: Cleans and processes raw scraped data
- **`build_dataset.py`**: Creates structured dataset for indexing

### Data Processing

- **`build_full_dataset.py`**: Combines all data sources into unified dataset
- **`embed_index.py`**: Creates FAISS vector index from passages
- **`test_retrieval.py`**: Tests and validates retrieval system

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/diyasaigal-PES1UG23AM101/afml-project.git
cd afml-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Running the Web Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

### Building the Dataset (Optional)

If you want to rebuild the corpus and index:

1. **Scrape Wikipedia articles**:
```bash
python diya/src/scrape_wiki.py --limit 100 --out data/raw/wiki
```

2. **Scrape news articles**:
```bash
python diya/src/scrape_news.py --urls diya/src/news_urls.txt --out data/raw/news
```

3. **Process raw data**:
```bash
python diya/src/process_raw.py
```

4. **Build full dataset**:
```bash
python src/build_full_dataset.py
```

5. **Create FAISS index**:
```bash
python src/embed_index.py
```

### Testing the System

Test basic functionality:
```bash
python test_pipeline.py
```

Test retrieval only:
```bash
python src/test_retrieval.py
```

### Evaluating Translation Quality

```bash
# Evaluate with sample data
python src/evaluate_translation.py --test-file data/sample_test_data.jsonl

# Evaluate RAG pipeline translations
python src/evaluate_translation.py \
    --rag-eval \
    --questions-file questions.txt \
    --ground-truth-file ground_truth.txt \
    --model gpt4
```

## Advanced Usage

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed documentation on:
- Fine-tuning translation models
- Advanced RAG pipeline configuration
- Batch processing
- Custom evaluation metrics
- Performance optimization

## Project Structure

```
afml-project/
├── app/
│   └── app.py                      # Enhanced Streamlit web application
├── diya/
│   └── src/                        # Data collection scripts
│       ├── scrape_wiki.py
│       ├── scrape_news.py
│       ├── add_bible.py
│       ├── process_raw.py
│       └── build_dataset.py
├── src/
│   ├── rag.py                      # Legacy RAG orchestration
│   ├── rag_pipeline.py             # Complete RAG pipeline with translation
│   ├── retriever.py                # FAISS-based retrieval
│   ├── reranker.py                 # Passage re-ranking
│   ├── generator.py                # OpenAI LLM generation
│   ├── prompts.py                  # Prompt templates
│   ├── evaluate_translation.py     # BLEU/METEOR evaluation
│   ├── build_full_dataset.py
│   ├── embed_index.py
│   └── test_retrieval.py
├── data/
│   └── sample_test_data.jsonl      # Sample test data for evaluation
├── notebooks/
│   └── embedding_evaluation.ipynb
├── test_pipeline.py                # Test suite
├── requirements.txt
├── USAGE_GUIDE.md                  # Detailed usage documentation
├── .env.example                    # Environment variable template
└── README.md
```

## Models Used

- **Embedding Model**: `sentence-transformers/LaBSE` - Multilingual sentence embeddings
- **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2` - Passage re-ranking
- **Vector Store**: FAISS - Efficient similarity search
- **LLM**: OpenAI GPT-4o-mini - Answer generation
- **Translation Models**:
  - `facebook/mbart-large-50-many-to-many-mmt` - Multilingual translation
  - `google/mt5-base` - Multilingual T5 for translation
  - OpenAI GPT-4 - High-quality translation via API

## Configuration

Key parameters in the RAG pipeline:

- **`top_k`**: Number of passages to retrieve (default: 10)
- **`rerank`**: Enable/disable re-ranking (default: True)
- **`temperature`**: LLM sampling temperature (default: 0.2)
- **`max_tokens`**: Maximum response length (default: 256)

## Data Sources

The Tulu corpus includes:
- Tulu Wikipedia articles
- Tulu news articles
- Bible content in Tulu
- Other curated Tulu language texts

## Contributing

This is a course project for AFML. For questions or contributions, please contact the project maintainers.

## License

This project is part of academic coursework at PES University.

## Acknowledgments

- Course: Advanced Foundations of Machine Learning (AFML)
- Institution: PES University
- Semester: 5

## Contact

- Repository: [afml-project](https://github.com/diyasaigal-PES1UG23AM101/afml-project)
- Owner: diyasaigal-PES1UG23AM101
