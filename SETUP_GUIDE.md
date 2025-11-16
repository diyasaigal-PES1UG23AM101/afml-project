# Setup Verification and Quick Start Commands

## First-Time Setup

### 1. Install Dependencies
```powershell
# Install all required packages
pip install -r requirements.txt

# Or install with specific versions
pip install --upgrade -r requirements.txt
```

### 2. Download NLTK Data (for BLEU/METEOR)
```powershell
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 3. Set Environment Variables
```powershell
# Set OpenAI API key
$env:OPENAI_API_KEY = "your-api-key-here"

# Optional: Create .env file
Copy-Item .env.example .env
# Then edit .env with your API key
```

### 4. Verify Installation
```powershell
# Run test suite
python test_pipeline.py

# Run workflow demo
python demo_workflow.py
```

---

## Quick Commands Reference

### Run Web Application
```powershell
streamlit run app/app.py
```

### Test Retrieval System
```powershell
python src/test_retrieval.py
```

### Evaluate Translations
```powershell
# Using sample data
python src/evaluate_translation.py --test-file data/sample_test_data.jsonl

# RAG evaluation mode
python src/evaluate_translation.py --rag-eval --questions-file questions.txt --ground-truth-file truth.txt --model gpt4
```

### Run Quick Examples
```powershell
# List examples
python quick_reference.py

# Run specific example (1-10)
python quick_reference.py 1
python quick_reference.py 4  # Translation evaluation
python quick_reference.py 8  # Batch processing
```

### Build Dataset and Index (if needed)
```powershell
# Scrape data
python diya/src/scrape_wiki.py --limit 50 --out data/raw/wiki
python diya/src/scrape_news.py --urls diya/src/news_urls.txt --out data/raw/news

# Process and build
python diya/src/process_raw.py
python src/build_full_dataset.py

# Create FAISS index
python src/embed_index.py
```

---

## Verification Checklist

### Check Python Version
```powershell
python --version  # Should be 3.8+
```

### Check Package Installation
```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import sentence_transformers; print('Sentence Transformers installed')"
python -c "import faiss; print('FAISS installed')"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import openai; print('OpenAI:', openai.__version__)"
python -c "import nltk; print('NLTK:', nltk.__version__)"
```

### Check API Key
```powershell
python -c "import os; print('API Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Check Project Structure
```powershell
Get-ChildItem -Recurse -Include *.py | Select-Object FullName
```

---

## Troubleshooting

### ImportError: No module named 'xyz'
```powershell
pip install xyz
# or
pip install -r requirements.txt --force-reinstall
```

### FAISS Not Found
```powershell
# For CPU-only systems
pip install faiss-cpu

# For GPU systems
pip install faiss-gpu
```

### NLTK Data Missing
```powershell
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
```

### OpenAI API Error
```powershell
# Verify API key is set
echo $env:OPENAI_API_KEY

# Set it if missing
$env:OPENAI_API_KEY = "sk-..."

# Check API key validity
python -c "import openai; import os; openai.api_key=os.getenv('OPENAI_API_KEY'); print(openai.Model.list())"
```

### Streamlit Not Running
```powershell
# Reinstall streamlit
pip install --upgrade streamlit

# Check firewall/port
streamlit run app/app.py --server.port 8502
```

### Out of Memory (for mBART/mT5)
```python
# Use smaller models
# Edit src/rag_pipeline.py:
TRANSLATION_MODELS["mt5"]["model_name"] = "google/mt5-small"

# Or reduce batch size in fine-tuning
pipeline.fine_tune_translator(..., batch_size=2)
```

---

## Common Workflows

### 1. Quick Test Run
```powershell
python test_pipeline.py
python demo_workflow.py
```

### 2. Launch Web App
```powershell
$env:OPENAI_API_KEY = "your-key"
streamlit run app/app.py
```

### 3. Evaluate Translation Quality
```powershell
python src/evaluate_translation.py --test-file data/sample_test_data.jsonl --output my_results.json
```

### 4. Run Custom Query
```powershell
python -c "from src.rag_pipeline import RAGPipeline; p = RAGPipeline(); r = p.query('What is Tulu?'); print(r['answer'])"
```

### 5. Batch Process Questions
```powershell
python quick_reference.py 8
```

---

## Performance Tips

### Speed Up Loading
```powershell
# Pre-load models by running once
python -c "from src.rag_pipeline import RAGPipeline; RAGPipeline()"
```

### Use GPU (if available)
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

### Optimize Retrieval
```python
# Use fewer passages
result = pipeline.query(question, top_k=5, rerank_top=3)

# Disable reranking for speed
pipeline = RAGPipeline(use_reranking=False)
```

---

## Development Commands

### Run Linter
```powershell
pip install flake8
flake8 src/ app/ --max-line-length=120
```

### Format Code
```powershell
pip install black
black src/ app/ --line-length=100
```

### Type Checking
```powershell
pip install mypy
mypy src/
```

### Generate Documentation
```powershell
pip install pdoc3
pdoc --html --output-dir docs src/
```

---

## Project Files Overview

| File | Purpose |
|------|---------|
| `src/rag_pipeline.py` | Main RAG implementation |
| `src/evaluate_translation.py` | BLEU/METEOR evaluation |
| `app/app.py` | Streamlit web interface |
| `test_pipeline.py` | Automated testing |
| `demo_workflow.py` | Complete workflow demo |
| `quick_reference.py` | Code examples |
| `requirements.txt` | Python dependencies |
| `README.md` | Project overview |
| `USAGE_GUIDE.md` | Detailed usage |
| `IMPLEMENTATION_STATUS.md` | Task completion |
| `PROJECT_SUMMARY.md` | Project summary |

---

## Support

For issues or questions:
1. Check USAGE_GUIDE.md
2. Run test_pipeline.py
3. Check error messages
4. Verify environment setup
5. Review documentation

---

**Quick Start: Run `python demo_workflow.py` to verify everything works!**
