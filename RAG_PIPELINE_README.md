# RAG Pipeline for Tulu-English Translation

This document describes the Retrieval-Augmented Generation (RAG) pipeline for translating Tulu text to English.

## Overview

The RAG pipeline combines:
1. **FAISS Vector Store**: Stores embeddings of Tulu passages for fast similarity search
2. **Embedding Model**: Multilingual sentence transformer for encoding text
3. **Translation Models**: mT5, mBART, or GPT-4 for translation
4. **Retrieval-Augmented Translation**: Uses retrieved context to improve translation quality

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Note: For GPU acceleration, you may want to install `faiss-gpu` instead of `faiss-cpu` and ensure PyTorch is installed with CUDA support.

## Usage

### 1. Build the FAISS Index

First, build the FAISS index from your passages file:

```bash
# Build from main passages file (includes all data)
python src/rag_pipeline.py --build_index --passages_file data/processed/passages.jsonl

# Or build from Bible passages (includes parallel English translations for better context)
python src/rag_pipeline.py --build_index --passages_file data/processed/bible_passages.jsonl
```

**Note**: Building from `bible_passages.jsonl` is recommended if you have parallel English translations, as it provides better context for RAG-based translation.

This will:
- Load all passages from the JSONL file
- Generate embeddings using the multilingual embedding model
- Build and save a FAISS index to `data/rag_index/`

### 2. Translate a Single Sentence

Translate a Tulu sentence to English:

```bash
# Using mT5
python src/rag_pipeline.py --translate "ಪ್ರಾರಂಭಿಸಿ" --model mT5 --k 3

# Using mBART
python src/rag_pipeline.py --translate "ಪ್ರಾರಂಭಿಸಿ" --model mBART --k 3

# Using GPT-4 (requires API key)
export OPENAI_API_KEY="your-api-key"
python src/rag_pipeline.py --translate "ಪ್ರಾರಂಭಿಸಿ" --model gpt4 --k 3
```

Parameters:
- `--translate`: The Tulu text to translate
- `--model`: Translation model (mT5, mBART, or gpt4)
- `--k`: Number of retrieved passages to use as context (default: 3)
- `--api_key`: GPT-4 API key (or set OPENAI_API_KEY environment variable)

### 3. Create a Test Set

Create a test set from passages with parallel English translations:

```bash
# Create test set from Bible passages (has parallel translations)
python src/create_test_set.py --input data/processed/bible_passages.jsonl --output data/test.jsonl --num_samples 100
```

The script will automatically look for passages with parallel English translations to create evaluation examples.

### 4. Evaluate Translation Quality

Evaluate the pipeline on a test set:

```bash
# Evaluate with a single model
python src/rag_pipeline.py --evaluate --test_file data/test.jsonl --model mT5 --k 3

# Evaluate with multiple models and k values
python src/evaluate_rag.py --test_file data/test.jsonl --models mT5 mBART --k_values 1 3 5
```

The evaluation will:
- Translate all test examples
- Calculate BLEU and METEOR scores
- Save translations to `data/evaluations/translations_*.jsonl`
- Save metrics to `data/evaluations/metrics_*.json`

## Architecture

### RAG Pipeline Components

1. **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
   - Multilingual sentence transformer
   - Supports 50+ languages including Tulu
   - Generates 384-dimensional embeddings

2. **FAISS Index**: 
   - L2 distance index for similarity search
   - Normalized embeddings for cosine similarity
   - Fast retrieval of top-k similar passages

3. **Translation Models**:
   - **mT5**: Google's multilingual T5 model
     - Prompt-based translation
     - Supports few-shot learning with retrieved context
   
   - **mBART**: Facebook's multilingual BART model
     - Many-to-many translation model
     - Supports 50 languages (Tulu may need proxy language)
   
   - **GPT-4**: OpenAI's GPT-4 API
     - Best translation quality
     - Requires API key and internet connection
     - Uses retrieved context in the prompt

### Retrieval-Augmented Translation

The pipeline retrieves similar Tulu passages and uses their English translations (if available) as context:

1. **Retrieval**: Given a Tulu query, retrieve top-k most similar passages
2. **Context Building**: Extract parallel English translations from retrieved passages
3. **Translation**: Use context to guide translation:
   - **mT5**: Few-shot examples in the prompt
   - **mBART**: Context used indirectly (model handles multilingual input)
   - **GPT-4**: Context included in the system prompt

## Evaluation Metrics

- **BLEU**: Bilingual Evaluation Understudy score (0-1, higher is better)
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering (0-1, higher is better)

## File Structure

```
data/
├── processed/
│   └── passages.jsonl          # Input passages
├── rag_index/
│   ├── faiss_index.index       # FAISS vector index
│   └── metadata.pkl            # Passage metadata
├── test.jsonl                  # Test set (created by create_test_set.py)
└── evaluations/
    ├── translations_*.jsonl    # Translation outputs
    └── metrics_*.json          # Evaluation metrics
```

## Configuration

### Embedding Model

Change the embedding model in `rag_pipeline.py`:

```python
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

Other options:
- `paraphrase-multilingual-mpnet-base-v2` (better quality, slower)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Translation Models

Available models are defined in `AVAILABLE_MODELS`:
- `mT5`: `google/mt5-base`
- `mBART`: `facebook/mbart-large-50-many-to-many-mmt`

### Index Parameters

- `k`: Number of retrieved passages (default: 3)
- `batch_size`: Batch size for embedding generation (default: 32)
- `max_context_length`: Maximum context length for translation (default: 512)

## Troubleshooting

### Index Not Found

If you get "Index not loaded" error:
```bash
python src/rag_pipeline.py --build_index --passages_file data/processed/passages.jsonl
```

### Out of Memory

If you run out of memory:
- Reduce `batch_size` in `build_index()`
- Use CPU instead of GPU: `--device cpu`
- Use a smaller embedding model

### Model Download Issues

Models are downloaded from HuggingFace on first use. Ensure you have:
- Internet connection
- Sufficient disk space (~2-5 GB per model)
- HuggingFace account (for some models)

### GPT-4 API Errors

- Ensure `OPENAI_API_KEY` is set
- Check your API quota
- Verify internet connection

## Example Output

### Translation Output

```
Tulu: ಪ್ರಾರಂಭಿಸಿ
English: Begin
```

### Evaluation Output

```
============================================================
EVALUATION RESULTS
============================================================
Model: mT5
Test examples: 100
Successfully translated: 100
Average BLEU: 0.4523
Average METEOR: 0.6234
============================================================
```

## Next Steps

1. **Fine-tuning**: Fine-tune mT5 or mBART on Tulu-English parallel data
2. **Better Embeddings**: Use domain-specific embeddings or fine-tune the embedding model
3. **Hybrid Retrieval**: Combine semantic search with keyword search
4. **Reranking**: Use a reranker to improve retrieved context quality
5. **Optimization**: Optimize translation accuracy by tuning k, context length, and model parameters

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)
- [METEOR Score](https://en.wikipedia.org/wiki/METEOR)

