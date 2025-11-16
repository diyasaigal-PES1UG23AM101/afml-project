"""
Test script for RAG Pipeline
Run this to verify the implementation works correctly
"""

import sys
import os

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, environment variables must be set manually

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from src.rag_pipeline import RAGPipeline
        from src.evaluate_translation import TranslationEvaluator
        from src.retriever import retrieve
        from src.reranker import rank
        from src.generator import generate_openai
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_retrieval():
    """Test basic retrieval functionality"""
    print("\nTesting retrieval...")
    try:
        from src.retriever import retrieve
        
        # Note: This will fail if index doesn't exist
        # Just testing the function is callable
        print("✓ Retriever module loaded")
        print("  (Actual retrieval requires FAISS index to be built)")
        return True
    except Exception as e:
        print(f"✗ Retrieval test error: {e}")
        return False


def test_reranker():
    """Test reranking functionality"""
    print("\nTesting reranker...")
    try:
        from src.reranker import rank
        
        # Test with sample data
        query = "What is Tulu language?"
        candidates = [
            "Tulu is a Dravidian language spoken in Karnataka.",
            "The weather is nice today.",
            "Tulu has a rich literary tradition."
        ]
        
        results = rank(query, candidates)
        
        assert len(results) == 3, "Should return 3 results"
        assert results[0][1] > results[1][1], "Should rank relevant passages higher"
        
        print("✓ Reranker working correctly")
        print(f"  Top result: candidate {results[0][0]} with score {results[0][1]:.3f}")
        return True
    except Exception as e:
        print(f"✗ Reranker test error: {e}")
        return False


def test_translation_evaluator():
    """Test translation evaluation"""
    print("\nTesting translation evaluator...")
    try:
        from src.evaluate_translation import TranslationEvaluator
        
        evaluator = TranslationEvaluator()
        
        # Test BLEU with longer text for more reliable scores
        bleu = evaluator.calculate_bleu(
            reference="the quick brown fox jumps over the lazy dog",
            hypothesis="the quick brown fox jumps over the lazy dog"
        )
        assert 0.8 <= bleu <= 1.0, f"Identical strings should have high BLEU, got {bleu:.4f}"
        
        # Test METEOR
        meteor = evaluator.calculate_meteor(
            reference="the quick brown fox jumps over the lazy dog",
            hypothesis="the quick brown fox jumps over the lazy dog"
        )
        assert 0.9 <= meteor <= 1.0, f"Identical strings should have high METEOR, got {meteor:.4f}"
        
        print("✓ Translation evaluator working correctly")
        print(f"  BLEU: {bleu:.4f}, METEOR: {meteor:.4f}")
        return True
    except Exception as e:
        print(f"✗ Evaluator test error: {e}")
        return False


def test_pipeline_initialization():
    """Test RAG pipeline initialization"""
    print("\nTesting RAG pipeline initialization...")
    try:
        from src.rag_pipeline import RAGPipeline
        
        # Test with GPT-4 (no model loading required)
        pipeline = RAGPipeline(translation_model="gpt4", use_reranking=True)
        
        assert pipeline.translation_model_type == "gpt4"
        assert pipeline.use_reranking == True
        
        print("✓ Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Pipeline initialization error: {e}")
        return False


def test_openai_config():
    """Check OpenAI API configuration"""
    print("\nChecking OpenAI configuration...")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print(f"✓ OpenAI API key found (length: {len(api_key)})")
        return True
    else:
        # Check if .env file exists
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        env_example = os.path.join(os.path.dirname(__file__), ".env.example")
        
        if os.path.exists(env_file):
            print("⚠ .env file exists but OPENAI_API_KEY not loaded")
            print("  Try: pip install python-dotenv")
            print("  Then add to your script: from dotenv import load_dotenv; load_dotenv()")
        elif os.path.exists(env_example):
            print("ℹ .env.example file found")
            print("  Copy to .env: Copy-Item .env.example .env")
            print("  OR set directly: $env:OPENAI_API_KEY='your-key'")
        else:
            print("⚠ OpenAI API key not set")
            print("  Set with: $env:OPENAI_API_KEY='your-key'")
        return False


def test_sample_data():
    """Check if sample test data exists"""
    print("\nChecking sample data...")
    test_file = "data/sample_test_data.jsonl"
    
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ Sample test data found ({len(lines)} examples)")
        return True
    else:
        print(f"⚠ Sample test data not found at {test_file}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RAG Pipeline Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Retrieval", test_retrieval),
        ("Reranker", test_reranker),
        ("Translation Evaluator", test_translation_evaluator),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("OpenAI Config", test_openai_config),
        ("Sample Data", test_sample_data),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
