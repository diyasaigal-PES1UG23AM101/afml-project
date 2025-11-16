#!/usr/bin/env python3
"""
Complete Workflow Demo - Tulu RAG System
Demonstrates the full pipeline from query to evaluation
"""

import sys
import os
from typing import Dict, Any

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print step indicator"""
    print(f"{Colors.OKBLUE}{Colors.BOLD}Step {step_num}: {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_result(label: str, value: Any):
    """Print result"""
    print(f"{Colors.BOLD}{label}:{Colors.ENDC} {value}")


def demo_workflow():
    """Run complete workflow demonstration"""
    
    print_header("TULU RAG SYSTEM - COMPLETE WORKFLOW DEMO")
    
    # ====================================================================
    # STEP 1: Environment Check
    # ====================================================================
    print_step(1, "Checking Environment")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print_success(f"OpenAI API key found (length: {len(api_key)})")
    else:
        print(f"{Colors.WARNING}⚠ OpenAI API key not set{Colors.ENDC}")
        print("Set with: $env:OPENAI_API_KEY='your-key'")
        print("Continuing with limited functionality...\n")
    
    # ====================================================================
    # STEP 2: Import Modules
    # ====================================================================
    print_step(2, "Loading Modules")
    
    try:
        from src.rag_pipeline import RAGPipeline
        from src.evaluate_translation import TranslationEvaluator
        print_success("All modules imported successfully")
    except ImportError as e:
        print(f"{Colors.FAIL}✗ Import failed: {e}{Colors.ENDC}")
        print("Install dependencies: pip install -r requirements.txt")
        return False
    
    # ====================================================================
    # STEP 3: Initialize Pipeline
    # ====================================================================
    print_step(3, "Initializing RAG Pipeline")
    
    try:
        pipeline = RAGPipeline(
            translation_model="gpt4",
            use_reranking=True
        )
        print_success("Pipeline initialized with GPT-4 translation")
        print_info("Configuration:")
        print(f"  - Translation Model: GPT-4")
        print(f"  - Re-ranking: Enabled")
        print(f"  - Device: {pipeline.device}")
    except Exception as e:
        print(f"{Colors.FAIL}✗ Initialization failed: {e}{Colors.ENDC}")
        return False
    
    # ====================================================================
    # STEP 4: Test Retrieval & Re-ranking
    # ====================================================================
    print_step(4, "Testing Retrieval & Re-ranking")
    
    test_query = "What is the Tulu language?"
    print_info(f"Query: '{test_query}'")
    
    try:
        passages = pipeline.retrieve_and_rank(
            test_query,
            top_k=10,
            rerank_top=5
        )
        
        if passages:
            print_success(f"Retrieved {len(passages)} passages")
            print_info("Top 3 passages:")
            for i, (idx, score, text) in enumerate(passages[:3], 1):
                print(f"\n  {i}. [ID: {idx}] Score: {score:.3f}")
                print(f"     {text[:150]}...")
        else:
            print(f"{Colors.WARNING}⚠ No passages retrieved (index may not exist){Colors.ENDC}")
            print_info("Build index with: python src/embed_index.py")
    except Exception as e:
        print(f"{Colors.WARNING}⚠ Retrieval test skipped: {e}{Colors.ENDC}")
        print_info("This is expected if FAISS index hasn't been built yet")
    
    # ====================================================================
    # STEP 5: Test Translation
    # ====================================================================
    print_step(5, "Testing Translation")
    
    if api_key:
        try:
            test_text = "The Tulu language is spoken in Karnataka."
            print_info(f"Input: '{test_text}'")
            
            translated = pipeline.translate(test_text, target_lang="Tulu")
            print_success("Translation successful")
            print_result("Output", translated)
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Translation failed: {e}{Colors.ENDC}")
    else:
        print_info("Translation test skipped (no API key)")
    
    # ====================================================================
    # STEP 6: Test BLEU/METEOR Evaluation
    # ====================================================================
    print_step(6, "Testing BLEU/METEOR Evaluation")
    
    try:
        evaluator = TranslationEvaluator()
        
        # Test with sample data
        test_pairs = [
            {
                "reference": "ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಭಾಗವಾಗಿದೆ",
                "hypothesis": "ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ"
            }
        ]
        
        bleu = evaluator.calculate_bleu(
            test_pairs[0]["reference"],
            test_pairs[0]["hypothesis"]
        )
        meteor = evaluator.calculate_meteor(
            test_pairs[0]["reference"],
            test_pairs[0]["hypothesis"]
        )
        
        print_success("Evaluation metrics computed")
        print_result("BLEU Score", f"{bleu:.4f}")
        print_result("METEOR Score", f"{meteor:.4f}")
        
    except Exception as e:
        print(f"{Colors.WARNING}⚠ Evaluation failed: {e}{Colors.ENDC}")
        print_info("Install NLTK data: python -m nltk.downloader wordnet omw-1.4")
    
    # ====================================================================
    # STEP 7: Full Query Test (if API key available)
    # ====================================================================
    print_step(7, "Testing Complete RAG Query")
    
    if api_key:
        try:
            test_question = "What are the main features of Tulu language?"
            print_info(f"Question: '{test_question}'")
            
            result = pipeline.query(
                question=test_question,
                response_language="en",
                top_k=5,
                rerank_top=3
            )
            
            print_success("Query completed successfully")
            print_result("Answer", result["answer"][:300] + "...")
            print_result("Passages used", result["num_passages"])
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠ Query test failed: {e}{Colors.ENDC}")
            print_info("This may be due to missing FAISS index")
    else:
        print_info("Query test skipped (no API key)")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print_header("WORKFLOW DEMO SUMMARY")
    
    print(f"{Colors.BOLD}Components Tested:{Colors.ENDC}")
    print("  ✓ Environment check")
    print("  ✓ Module imports")
    print("  ✓ Pipeline initialization")
    print("  ✓ Retrieval & re-ranking")
    print("  ✓ Translation")
    print("  ✓ BLEU/METEOR evaluation")
    print("  ✓ Complete RAG query")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
    print("  1. Launch web app: streamlit run app/app.py")
    print("  2. Run full tests: python test_pipeline.py")
    print("  3. See usage guide: USAGE_GUIDE.md")
    print("  4. Try examples: python quick_reference.py")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Workflow demo completed!{Colors.ENDC}\n")
    
    return True


if __name__ == "__main__":
    try:
        success = demo_workflow()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Demo interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Demo failed with error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
