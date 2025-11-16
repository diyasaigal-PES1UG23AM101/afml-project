# Tulu RAG System - Documentation Index

Welcome! This document helps you navigate all the project documentation.

---

## 📚 Documentation Files

### 🚀 Quick Start
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation and setup instructions
  - First-time setup steps
  - Verification commands
  - Troubleshooting tips
  - Common workflows

### 📖 Main Documentation
- **[README.md](README.md)** - Project overview and introduction
  - Project description
  - Features list
  - Architecture overview
  - Quick installation
  - Basic usage examples

### 📘 Detailed Guides
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Comprehensive usage documentation
  - Basic usage examples
  - Fine-tuning guide
  - Evaluation workflows
  - Model comparison
  - Performance tips
  - API reference

### ✅ Project Status
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Task completion checklist
  - All implemented features
  - Task-by-task breakdown
  - Technical specifications
  - Usage examples for each component

### 📊 Project Summary
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - High-level project summary
  - Achievement summary
  - Key features
  - Demonstration guide
  - Future enhancements

---

## 💻 Code Files

### Main Implementation
- **`src/rag_pipeline.py`** - Complete RAG pipeline with translation
  - `RAGPipeline` class
  - Multi-model translation support
  - Fine-tuning capabilities
  - Query processing

- **`src/evaluate_translation.py`** - BLEU/METEOR evaluation
  - `TranslationEvaluator` class
  - Metrics computation
  - Batch evaluation
  - Result reporting

- **`app/app.py`** - Streamlit web application
  - Interactive UI
  - Model configuration
  - Real-time queries

### Supporting Code
- **`src/retriever.py`** - FAISS-based retrieval
- **`src/reranker.py`** - Cross-encoder re-ranking
- **`src/generator.py`** - OpenAI LLM generation
- **`src/prompts.py`** - Prompt templates
- **`src/rag.py`** - Legacy RAG implementation (backward compatible)

### Testing & Examples
- **`test_pipeline.py`** - Automated test suite
- **`demo_workflow.py`** - Complete workflow demonstration
- **`quick_reference.py`** - Code examples (10+ examples)

---

## 🎯 Where to Start?

### New to the Project?
1. Read **[README.md](README.md)** - Get project overview
2. Follow **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Set up environment
3. Run `python test_pipeline.py` - Verify installation
4. Run `python demo_workflow.py` - See it in action
5. Launch `streamlit run app/app.py` - Try the web interface

### Want to Use the System?
1. Check **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed examples
2. Run `python quick_reference.py` - See code examples
3. Try examples: `python quick_reference.py 1` through `10`

### Want to Understand Implementation?
1. Read **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - See what was built
2. Review **`src/rag_pipeline.py`** - Main implementation
3. Check **`src/evaluate_translation.py`** - Evaluation code

### Need to Demonstrate?
1. Review **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Demo guide
2. Run `streamlit run app/app.py` - Live demo
3. Run `python src/evaluate_translation.py --test-file data/sample_test_data.jsonl` - Show metrics

---

## 📋 Quick Reference by Task

### Installation & Setup
- **Setup instructions**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Dependencies**: `requirements.txt`
- **Environment config**: `.env.example`

### Running the System
- **Web interface**: `streamlit run app/app.py`
- **Command line**: Use code from `quick_reference.py`
- **Testing**: `python test_pipeline.py`

### Translation & Evaluation
- **Translation guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Translation Models section
- **Evaluation guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Evaluating Translation Quality
- **Example code**: `quick_reference.py` - Examples 4, 5

### Fine-tuning
- **Fine-tuning guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md) - Fine-Tuning Translation Models
- **Example code**: `quick_reference.py` - Example 7

### Development
- **Architecture**: [README.md](README.md) - Architecture section
- **API docs**: Code docstrings in `src/` files
- **Testing**: `test_pipeline.py`

---

## 🔍 Finding Information

### How do I...

**...install the system?**
→ [SETUP_GUIDE.md](SETUP_GUIDE.md) - First-Time Setup

**...run a query?**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Basic Usage
→ `quick_reference.py` - Examples 1, 2

**...translate text?**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Translation Only
→ `quick_reference.py` - Example 3

**...evaluate translations?**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Evaluating Translation Quality
→ `quick_reference.py` - Examples 4, 5

**...fine-tune a model?**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Fine-Tuning Translation Models
→ `quick_reference.py` - Example 7

**...use different translation models?**
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Translation Model Comparison
→ `quick_reference.py` - Example 6

**...process multiple questions?**
→ `quick_reference.py` - Example 8

**...compare retrieval methods?**
→ `quick_reference.py` - Example 9

**...create custom prompts?**
→ `quick_reference.py` - Example 10

**...troubleshoot issues?**
→ [SETUP_GUIDE.md](SETUP_GUIDE.md) - Troubleshooting
→ [USAGE_GUIDE.md](USAGE_GUIDE.md) - Troubleshooting

---

## 📦 File Organization

```
Documentation:
├── README.md                     # Project overview (START HERE)
├── SETUP_GUIDE.md               # Installation & setup
├── USAGE_GUIDE.md               # Detailed usage guide
├── IMPLEMENTATION_STATUS.md     # Task completion status
├── PROJECT_SUMMARY.md           # Project summary
└── DOCS_INDEX.md                # This file

Code:
├── src/
│   ├── rag_pipeline.py          # Main RAG implementation
│   ├── evaluate_translation.py  # Evaluation suite
│   ├── retriever.py             # FAISS retrieval
│   ├── reranker.py              # Re-ranking
│   ├── generator.py             # LLM generation
│   └── prompts.py               # Prompt templates
├── app/
│   └── app.py                   # Streamlit web app
├── test_pipeline.py             # Test suite
├── demo_workflow.py             # Workflow demo
└── quick_reference.py           # Code examples

Configuration:
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
└── .gitignore                   # Git exclusions

Data:
└── data/
    └── sample_test_data.jsonl   # Sample test data
```

---

## 🎓 Learning Path

### Beginner (Just getting started)
1. Read [README.md](README.md)
2. Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)
3. Run `python demo_workflow.py`
4. Launch web app: `streamlit run app/app.py`

### Intermediate (Using the system)
1. Study [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Run examples from `quick_reference.py`
3. Experiment with different models
4. Try custom queries

### Advanced (Development & customization)
1. Review [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
2. Study source code in `src/`
3. Implement custom features
4. Fine-tune models

---

## 📞 Quick Help

### Something's not working?
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) - Troubleshooting
2. Run `python test_pipeline.py` to diagnose
3. Verify environment: `python demo_workflow.py`

### Need a specific feature?
1. Search [USAGE_GUIDE.md](USAGE_GUIDE.md)
2. Check `quick_reference.py` examples
3. Review source code docstrings

### Want to contribute?
1. Read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
2. Review code in `src/`
3. Follow existing patterns

---

## ✨ Key Highlights

### What Makes This Project Special?
✅ Complete RAG pipeline with multiple translation models  
✅ Comprehensive evaluation suite (BLEU/METEOR)  
✅ Production-ready web interface  
✅ Extensive documentation  
✅ Working code examples  
✅ Test suite included  

### All Checklist Items Completed!
✅ RAG architecture built  
✅ FAISS + embeddings loaded  
✅ LLM integration (GPT-4, mBART, mT5)  
✅ Retrieval-augmented prompts  
✅ Fine-tuning capability  
✅ BLEU/METEOR evaluation  

---

## 🚀 Ready to Start?

Run this to verify everything works:
```bash
python demo_workflow.py
```

Then launch the web app:
```bash
streamlit run app/app.py
```

**Happy coding!** 🎉
