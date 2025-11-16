# RAG Pipeline Usage Guide

This guide explains how to use the complete RAG pipeline with translation capabilities.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file or set environment variable:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"

# Linux/Mac
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Web App

```bash
streamlit run app/app.py
```

Visit `http://localhost:8501` in your browser.

## Using the RAG Pipeline Programmatically

### Basic Usage

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    translation_model="gpt4",  # or "mbart" or "mt5"
    use_reranking=True
)

# Ask a question
result = pipeline.query(
    question="What is the history of Tulu language?",
    response_language="en",  # "en", "tulu", or "both"
    top_k=10,
    rerank_top=5
)

print("Answer:", result["answer"])
print(f"\nFound {len(result['passages'])} relevant passages")
```

### Translation Only

```python
# Translate English to Tulu
tulu_text = pipeline.translate(
    "The Tulu language is a Dravidian language.",
    target_lang="Tulu"
)
print(tulu_text)
```

### Advanced Query Options

```python
# Query with question translation (for better cross-lingual retrieval)
result = pipeline.query(
    question="ತುಳು ಭಾಷೆಯ ಇತಿಹಾಸ ಏನು?",  # Question in Tulu
    response_language="tulu",
    translate_question=True,  # Translates to English for retrieval
    top_k=15,
    rerank_top=7
)
```

## Fine-Tuning Translation Models

### Prepare Training Data

Create a JSONL file with parallel sentences:

```jsonl
{"source": "Hello, how are you?", "target": "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?"}
{"source": "The weather is nice today.", "target": "ಇಂದು ಹವಾಮಾನ ಚೆನ್ನಾಗಿದೆ."}
```

### Fine-Tune

```python
from src.rag_pipeline import RAGPipeline

# Initialize with mBART or mT5
pipeline = RAGPipeline(translation_model="mbart")

# Fine-tune
pipeline.fine_tune_translator(
    train_data_path="data/translation_pairs.jsonl",
    output_dir="models/fine_tuned_mbart",
    num_epochs=3,
    batch_size=8,
    learning_rate=5e-5
)
```

### Load Fine-Tuned Model

After fine-tuning, update the model path in `rag_pipeline.py`:

```python
TRANSLATION_MODELS = {
    "mbart": {
        "model_name": "models/fine_tuned_mbart",  # Updated path
        # ...
    }
}
```

## Evaluating Translation Quality

### Prepare Test Data

Create files with questions and ground truth translations:

**questions.txt:**
```
What is the capital of Karnataka?
Tell me about Tulu cuisine.
```

**ground_truth.txt:**
```
ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.
ತುಳು ಪಾಕಪದ್ಧತಿ ಬಹಳ ವೈವಿಧ್ಯಮಯವಾಗಿದೆ.
```

### Run Evaluation

```bash
python src/evaluate_translation.py \
    --rag-eval \
    --questions-file questions.txt \
    --ground-truth-file ground_truth.txt \
    --model gpt4 \
    --output results.json
```

### Evaluate Pre-Generated Translations

If you have pre-generated translations in JSONL format:

**test_translations.jsonl:**
```jsonl
{"reference": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು.", "hypothesis": "ಕರ್ನಾಟಕ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು ಆಗಿದೆ."}
```

```bash
python src/evaluate_translation.py \
    --test-file test_translations.jsonl \
    --output eval_results.json
```

### Programmatic Evaluation

```python
from src.evaluate_translation import TranslationEvaluator

evaluator = TranslationEvaluator()

# Single pair
bleu = evaluator.calculate_bleu(
    reference="ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷೆ",
    hypothesis="ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷೆಯಾಗಿದೆ"
)
meteor = evaluator.calculate_meteor(
    reference="ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷೆ",
    hypothesis="ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷೆಯಾಗಿದೆ"
)

print(f"BLEU: {bleu:.4f}, METEOR: {meteor:.4f}")

# Multiple pairs
test_data = [
    {"reference": "...", "hypothesis": "..."},
    {"reference": "...", "hypothesis": "..."}
]

results = evaluator.evaluate_translation_pairs(test_data)
evaluator.print_summary(results)
```

## Translation Model Comparison

### GPT-4 (Recommended)
- **Pros:** Best quality, handles Tulu well, no local setup
- **Cons:** API costs, requires internet
- **Use when:** Quality is priority, budget allows

### mBART
- **Pros:** Open source, multilingual, can fine-tune
- **Cons:** Tulu not directly supported (uses Kannada proxy), larger model
- **Use when:** Need offline capability, want to fine-tune

### mT5
- **Pros:** Open source, flexible, good for low-resource languages
- **Cons:** Requires careful prompting, moderate quality
- **Use when:** Experimenting, need flexibility

## Performance Tips

### Optimize Retrieval

```python
# Use fewer passages for faster responses
result = pipeline.query(question, top_k=5, rerank_top=3)

# Disable reranking for speed
pipeline = RAGPipeline(use_reranking=False)
```

### Batch Processing

```python
questions = ["Q1", "Q2", "Q3", ...]
results = []

for q in questions:
    result = pipeline.query(q, response_language="en")
    results.append(result)
```

### GPU Acceleration

For mBART/mT5 models:

```python
import torch

pipeline = RAGPipeline(
    translation_model="mbart",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

## Troubleshooting

### FAISS Index Not Found

```bash
# Build the index first
python src/embed_index.py
```

### OpenAI API Errors

```python
# Check API key
import os
print(os.getenv("OPENAI_API_KEY"))

# Update to latest OpenAI library
pip install --upgrade openai
```

### NLTK Data Missing

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Translation Model Loading Issues

```python
# Use smaller models
TRANSLATION_MODELS["mt5"]["model_name"] = "google/mt5-small"

# Clear cache if corrupted
import shutil
shutil.rmtree("~/.cache/huggingface/transformers")
```

## Example Workflows

### Complete Question-Answer Pipeline

```python
from src.rag_pipeline import RAGPipeline

# 1. Initialize
pipeline = RAGPipeline(translation_model="gpt4", use_reranking=True)

# 2. Query
result = pipeline.query(
    question="What are the main features of Tulu literature?",
    response_language="both",  # Get both English and Tulu
    top_k=10
)

# 3. Process results
print("English + Tulu Answer:")
print(result["answer"])

print("\nSource passages:")
for idx, score, text in result["passages"][:3]:
    print(f"[{idx}] Score: {score:.3f}")
    print(text[:200], "...\n")
```

### Build Custom Dataset → Index → Query

```bash
# 1. Collect data
python diya/src/scrape_wiki.py --limit 50
python diya/src/scrape_news.py --urls diya/src/news_urls.txt

# 2. Process
python diya/src/process_raw.py
python src/build_full_dataset.py

# 3. Index
python src/embed_index.py

# 4. Query
python -c "from src.rag_pipeline import RAGPipeline; \
           p = RAGPipeline(); \
           r = p.query('What is Tulu?'); \
           print(r['answer'])"
```

## API Reference

See individual files for detailed API documentation:
- `src/rag_pipeline.py` - Main RAG pipeline
- `src/evaluate_translation.py` - Evaluation metrics
- `src/retriever.py` - Vector retrieval
- `src/reranker.py` - Passage reranking
- `src/generator.py` - LLM generation
