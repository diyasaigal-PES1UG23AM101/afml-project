"""
Quick Reference Script - Common RAG Pipeline Operations
Copy and paste these examples for quick testing
"""

# ============================================================
# EXAMPLE 1: Basic Query with GPT-4 Translation
# ============================================================
def example_basic_query():
    from src.rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline(translation_model="gpt4", use_reranking=True)
    
    # Ask a question
    result = pipeline.query(
        question="What is the history of Tulu language?",
        response_language="en",
        top_k=10,
        rerank_top=5
    )
    
    print("ANSWER:", result["answer"])
    print(f"\nUsed {len(result['passages'])} passages")


# ============================================================
# EXAMPLE 2: Bilingual Response (English + Tulu)
# ============================================================
def example_bilingual():
    from src.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline(translation_model="gpt4")
    
    result = pipeline.query(
        question="What are the main features of Tulu literature?",
        response_language="both",  # Get both languages
        top_k=10
    )
    
    print(result["answer"])


# ============================================================
# EXAMPLE 3: Translate Text Only
# ============================================================
def example_translation_only():
    from src.rag_pipeline import RAGPipeline
    
    pipeline = RAGPipeline(translation_model="gpt4")
    
    english_text = "The Tulu language is a Dravidian language spoken in Karnataka."
    tulu_text = pipeline.translate(english_text, target_lang="Tulu")
    
    print("English:", english_text)
    print("Tulu:", tulu_text)


# ============================================================
# EXAMPLE 4: Evaluate Translation Quality
# ============================================================
def example_evaluation():
    from src.evaluate_translation import TranslationEvaluator
    
    evaluator = TranslationEvaluator()
    
    # Single pair evaluation
    bleu = evaluator.calculate_bleu(
        reference="ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಭಾಗವಾಗಿದೆ",
        hypothesis="ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ"
    )
    
    meteor = evaluator.calculate_meteor(
        reference="ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಭಾಗವಾಗಿದೆ",
        hypothesis="ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ"
    )
    
    print(f"BLEU Score: {bleu:.4f}")
    print(f"METEOR Score: {meteor:.4f}")


# ============================================================
# EXAMPLE 5: Evaluate from Test File
# ============================================================
def example_file_evaluation():
    from src.evaluate_translation import TranslationEvaluator, load_test_data
    
    evaluator = TranslationEvaluator()
    
    # Load test data
    test_data = load_test_data("data/sample_test_data.jsonl")
    
    # Evaluate
    results = evaluator.evaluate_translation_pairs(
        test_data,
        save_results=True,
        output_path="my_eval_results.json"
    )
    
    # Print summary
    evaluator.print_summary(results)


# ============================================================
# EXAMPLE 6: Use mBART Translation Model
# ============================================================
def example_mbart_translation():
    from src.rag_pipeline import RAGPipeline
    
    # Initialize with mBART (will download model first time)
    pipeline = RAGPipeline(translation_model="mbart", use_reranking=True)
    
    # Query
    result = pipeline.query(
        question="Tell me about Tulu cuisine",
        response_language="tulu",
        top_k=10
    )
    
    print(result["answer"])


# ============================================================
# EXAMPLE 7: Fine-tune Translation Model
# ============================================================
def example_fine_tuning():
    """
    NOTE: Requires training data in JSONL format:
    {"source": "English text", "target": "Tulu translation"}
    """
    from src.rag_pipeline import RAGPipeline
    
    # Initialize with base model
    pipeline = RAGPipeline(translation_model="mbart")
    
    # Fine-tune (requires GPU for best performance)
    pipeline.fine_tune_translator(
        train_data_path="data/my_training_data.jsonl",
        output_dir="models/fine_tuned_tulu_mbart",
        num_epochs=3,
        batch_size=4,  # Reduce if out of memory
        learning_rate=5e-5
    )
    
    print("Fine-tuning complete!")


# ============================================================
# EXAMPLE 8: Batch Processing Multiple Questions
# ============================================================
def example_batch_processing():
    from src.rag_pipeline import RAGPipeline
    import json
    
    pipeline = RAGPipeline(translation_model="gpt4")
    
    questions = [
        "What is the Tulu script called?",
        "Where is Tulu spoken?",
        "What is the history of Tulu literature?",
    ]
    
    results = []
    for i, q in enumerate(questions, 1):
        print(f"Processing {i}/{len(questions)}: {q}")
        result = pipeline.query(q, response_language="en", top_k=5)
        results.append({
            "question": q,
            "answer": result["answer"],
            "num_passages": result["num_passages"]
        })
    
    # Save results
    with open("batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {len(results)} questions. Results saved to batch_results.json")


# ============================================================
# EXAMPLE 9: Test Retrieval and Reranking
# ============================================================
def example_retrieval_comparison():
    from src.rag_pipeline import RAGPipeline
    
    question = "What are Tulu festivals?"
    
    # Without reranking
    pipeline_no_rerank = RAGPipeline(use_reranking=False)
    result_no_rerank = pipeline_no_rerank.retrieve_and_rank(question, top_k=10, rerank_top=5)
    
    # With reranking
    pipeline_rerank = RAGPipeline(use_reranking=True)
    result_rerank = pipeline_rerank.retrieve_and_rank(question, top_k=10, rerank_top=5)
    
    print("Top 3 without reranking:")
    for i, (idx, score, text) in enumerate(result_no_rerank[:3], 1):
        print(f"{i}. Score: {score:.3f} - {text[:100]}...")
    
    print("\nTop 3 with reranking:")
    for i, (idx, score, text) in enumerate(result_rerank[:3], 1):
        print(f"{i}. Score: {score:.3f} - {text[:100]}...")


# ============================================================
# EXAMPLE 10: Custom Prompt Testing
# ============================================================
def example_custom_generation():
    from src.rag_pipeline import RAGPipeline
    from src.generator import generate_openai
    
    pipeline = RAGPipeline()
    
    # Get passages
    passages = pipeline.retrieve_and_rank(
        "What is Tulu cuisine?",
        top_k=10,
        rerank_top=3
    )
    
    # Format passages
    passages_text = pipeline.format_passages_for_prompt(passages)
    
    # Custom prompt
    custom_prompt = f"""Based on these passages, write a brief paragraph about Tulu cuisine.
    Be informative but concise.
    
    Passages:
    {passages_text}
    
    Paragraph:"""
    
    answer = generate_openai(custom_prompt, max_tokens=200, temperature=0.3)
    print(answer)


# ============================================================
# RUN EXAMPLES
# ============================================================
if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic Query", example_basic_query),
        "2": ("Bilingual Response", example_bilingual),
        "3": ("Translation Only", example_translation_only),
        "4": ("Evaluate Translation", example_evaluation),
        "5": ("File Evaluation", example_file_evaluation),
        "6": ("mBART Translation", example_mbart_translation),
        "7": ("Fine-tuning", example_fine_tuning),
        "8": ("Batch Processing", example_batch_processing),
        "9": ("Retrieval Comparison", example_retrieval_comparison),
        "10": ("Custom Prompts", example_custom_generation),
    }
    
    print("="*60)
    print("RAG Pipeline - Quick Reference Examples")
    print("="*60)
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    print("="*60)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}\n")
            try:
                func()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Invalid choice: {choice}")
    else:
        print("\nUsage: python quick_reference.py <example_number>")
        print("Example: python quick_reference.py 1")
