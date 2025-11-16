"""
Translation Evaluation with BLEU and METEOR scores
Evaluates translation quality for the RAG system
"""

import json
import argparse
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

# BLEU and METEOR imports
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    NLTK_AVAILABLE = True
except ImportError:
    print("Warning: NLTK not installed. Install with: pip install nltk")
    NLTK_AVAILABLE = False

from src.rag_pipeline import RAGPipeline


class TranslationEvaluator:
    """
    Evaluates translation quality using BLEU and METEOR metrics
    """
    
    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for evaluation. Install with: pip install nltk")
        self.smoothing = SmoothingFunction()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization"""
        return text.lower().split()
    
    def calculate_bleu(
        self,
        reference: str,
        hypothesis: str,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> float:
        """
        Calculate BLEU score for a single sentence
        
        Args:
            reference: Ground truth translation
            hypothesis: Generated translation
            weights: N-gram weights (default: equal weights for 1-4 grams)
        
        Returns:
            BLEU score (0-1)
        """
        ref_tokens = [self.tokenize(reference)]
        hyp_tokens = self.tokenize(hypothesis)
        
        # Use smoothing to avoid zero scores
        score = sentence_bleu(
            ref_tokens,
            hyp_tokens,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        return score
    
    def calculate_meteor(self, reference: str, hypothesis: str) -> float:
        """
        Calculate METEOR score for a single sentence
        
        Args:
            reference: Ground truth translation
            hypothesis: Generated translation
        
        Returns:
            METEOR score (0-1)
        """
        ref_tokens = self.tokenize(reference)
        hyp_tokens = self.tokenize(hypothesis)
        
        score = meteor_score([ref_tokens], hyp_tokens)
        return score
    
    def calculate_corpus_bleu(
        self,
        references: List[str],
        hypotheses: List[str],
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> float:
        """
        Calculate BLEU score over entire corpus
        
        Args:
            references: List of ground truth translations
            hypotheses: List of generated translations
            weights: N-gram weights
        
        Returns:
            Corpus-level BLEU score
        """
        refs = [[self.tokenize(ref)] for ref in references]
        hyps = [self.tokenize(hyp) for hyp in hypotheses]
        
        score = corpus_bleu(
            refs,
            hyps,
            weights=weights,
            smoothing_function=self.smoothing.method1
        )
        return score
    
    def evaluate_translation_pairs(
        self,
        test_data: List[Dict[str, str]],
        save_results: bool = True,
        output_path: str = "evaluation_results.json"
    ) -> Dict[str, any]:
        """
        Evaluate a dataset of translation pairs
        
        Args:
            test_data: List of dicts with "reference" and "hypothesis" keys
            save_results: Whether to save detailed results
            output_path: Where to save results
        
        Returns:
            Dictionary with evaluation metrics
        """
        bleu_scores = []
        meteor_scores = []
        detailed_results = []
        
        for i, item in enumerate(test_data):
            ref = item["reference"]
            hyp = item["hypothesis"]
            
            # Calculate scores
            bleu = self.calculate_bleu(ref, hyp)
            meteor = self.calculate_meteor(ref, hyp)
            
            bleu_scores.append(bleu)
            meteor_scores.append(meteor)
            
            detailed_results.append({
                "index": i,
                "reference": ref,
                "hypothesis": hyp,
                "bleu": bleu,
                "meteor": meteor
            })
            
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i + 1}/{len(test_data)} pairs...")
        
        # Corpus-level BLEU
        corpus_bleu_score = self.calculate_corpus_bleu(
            [item["reference"] for item in test_data],
            [item["hypothesis"] for item in test_data]
        )
        
        # Aggregate results
        results = {
            "num_samples": len(test_data),
            "average_bleu": np.mean(bleu_scores),
            "median_bleu": np.median(bleu_scores),
            "std_bleu": np.std(bleu_scores),
            "corpus_bleu": corpus_bleu_score,
            "average_meteor": np.mean(meteor_scores),
            "median_meteor": np.median(meteor_scores),
            "std_meteor": np.std(meteor_scores),
            "detailed_results": detailed_results if save_results else None
        }
        
        # Save results
        if save_results:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {output_path}")
        
        return results
    
    def evaluate_rag_translations(
        self,
        questions: List[str],
        ground_truth_translations: List[str],
        translation_model: str = "gpt4",
        output_path: str = "rag_translation_eval.json"
    ) -> Dict[str, any]:
        """
        Evaluate RAG pipeline translations against ground truth
        
        Args:
            questions: List of questions in English
            ground_truth_translations: Expected translations in Tulu
            translation_model: Which model to use ("mbart", "mt5", "gpt4")
            output_path: Where to save results
        
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating RAG translations using {translation_model}...")
        
        pipeline = RAGPipeline(translation_model=translation_model, use_reranking=True)
        
        test_data = []
        
        for i, (question, ground_truth) in enumerate(zip(questions, ground_truth_translations)):
            print(f"\nProcessing {i+1}/{len(questions)}: {question[:50]}...")
            
            # Get RAG answer in English
            result = pipeline.query(question, response_language="en", top_k=10)
            english_answer = result["answer"]
            
            # Translate to Tulu
            translated_answer = pipeline.translate(english_answer, target_lang="Tulu")
            
            test_data.append({
                "reference": ground_truth,
                "hypothesis": translated_answer,
                "question": question,
                "english_answer": english_answer
            })
        
        # Evaluate
        results = self.evaluate_translation_pairs(test_data, save_results=True, output_path=output_path)
        
        return results
    
    def print_summary(self, results: Dict[str, any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("TRANSLATION EVALUATION SUMMARY")
        print("="*60)
        print(f"Number of samples: {results['num_samples']}")
        print(f"\nBLEU Scores:")
        print(f"  Average:  {results['average_bleu']:.4f}")
        print(f"  Median:   {results['median_bleu']:.4f}")
        print(f"  Std Dev:  {results['std_bleu']:.4f}")
        print(f"  Corpus:   {results['corpus_bleu']:.4f}")
        print(f"\nMETEOR Scores:")
        print(f"  Average:  {results['average_meteor']:.4f}")
        print(f"  Median:   {results['median_meteor']:.4f}")
        print(f"  Std Dev:  {results['std_meteor']:.4f}")
        print("="*60)


def load_test_data(file_path: str) -> List[Dict[str, str]]:
    """Load test data from JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate translation quality")
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to test data JSONL file with 'reference' and 'hypothesis' fields"
    )
    parser.add_argument(
        "--rag-eval",
        action="store_true",
        help="Evaluate RAG pipeline translations"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        help="File with questions (one per line) for RAG evaluation"
    )
    parser.add_argument(
        "--ground-truth-file",
        type=str,
        help="File with ground truth Tulu translations (one per line)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
        choices=["mbart", "mt5", "gpt4"],
        help="Translation model to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    evaluator = TranslationEvaluator()
    
    if args.rag_eval:
        # Evaluate RAG pipeline
        if not args.questions_file or not args.ground_truth_file:
            print("Error: --questions-file and --ground-truth-file required for RAG evaluation")
            return
        
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
        
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = [line.strip() for line in f if line.strip()]
        
        if len(questions) != len(ground_truth):
            print(f"Error: Mismatch in number of questions ({len(questions)}) and ground truth ({len(ground_truth)})")
            return
        
        results = evaluator.evaluate_rag_translations(
            questions,
            ground_truth,
            translation_model=args.model,
            output_path=args.output
        )
        evaluator.print_summary(results)
    
    elif args.test_file:
        # Evaluate pre-generated translations
        test_data = load_test_data(args.test_file)
        results = evaluator.evaluate_translation_pairs(test_data, save_results=True, output_path=args.output)
        evaluator.print_summary(results)
    
    else:
        # Demo evaluation
        print("Running demo evaluation...")
        demo_data = [
            {
                "reference": "ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಭಾಗವಾಗಿದೆ",
                "hypothesis": "ತುಳು ಭಾಷೆಯು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ"
            },
            {
                "reference": "ಕರ್ನಾಟಕ ರಾಜ್ಯದಲ್ಲಿ ತುಳು ಮಾತನಾಡುತ್ತಾರೆ",
                "hypothesis": "ಕರ್ನಾಟಕದಲ್ಲಿ ಜನರು ತುಳು ಮಾತನಾಡುತ್ತಾರೆ"
            }
        ]
        
        results = evaluator.evaluate_translation_pairs(demo_data, save_results=True, output_path="demo_eval.json")
        evaluator.print_summary(results)


if __name__ == "__main__":
    main()
