# Quick Start Guide - RAG Pipeline

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Build the Index

```bash
# Recommended: Use Bible passages with parallel translations
python src/rag_pipeline.py --build_index --passages_file data/processed/bible_passages.jsonl
```

## Step 3: Translate

```bash
# Translate a Tulu sentence
python src/rag_pipeline.py --translate "ಪ್ರಾರಂಭಿಸಿ" --model mT5 --k 3
```

## Step 4: Evaluate (Optional)

```bash
# Create test set
python src/create_test_set.py --input data/processed/bible_passages.jsonl --output data/test.jsonl --num_samples 100

# Evaluate
python src/rag_pipeline.py --evaluate --test_file data/test.jsonl --model mT5 --k 3
```

## What's Included

- ✅ **rag_pipeline.py**: Main RAG pipeline with FAISS, embeddings, and translation models
- ✅ **create_test_set.py**: Script to create test sets from parallel translations
- ✅ **evaluate_rag.py**: Convenience script for multi-model evaluation
- ✅ **RAG_PIPELINE_README.md**: Complete documentation

## Models Supported

- **mT5**: Google's multilingual T5 (good balance of quality and speed)
- **mBART**: Facebook's multilingual BART (many-to-many translation)
- **GPT-4**: OpenAI's GPT-4 API (best quality, requires API key)

## Features

- Retrieval-augmented translation using FAISS
- Multilingual embeddings for Tulu text
- BLEU and METEOR evaluation metrics
- Support for parallel translation context
- Flexible model selection

## Next Steps

See `RAG_PIPELINE_README.md` for detailed documentation and advanced usage.

