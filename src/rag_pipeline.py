"""
Complete RAG Pipeline Implementation
- Integrates retrieval, reranking, and generation
- Supports translation with mT5, mBART, or GPT-4 API
- Implements retrieval-augmented prompts
- Includes fine-tuning capability
"""

import os
import json
from typing import List, Tuple, Dict, Optional
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch

from src.retriever import retrieve
from src.reranker import rank
from src.generator import generate_openai

# Translation model configurations
TRANSLATION_MODELS = {
    "mbart": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "src_lang": "en_XX",  # English
        "tgt_lang": "kn_IN",  # Kannada (closest to Tulu)
    },
    "mt5": {
        "model_name": "google/mt5-small",  # Smaller, faster download (~300MB)
    }
}

class RAGPipeline:
    """
    Full RAG Pipeline with translation support
    """
    
    def __init__(
        self,
        translation_model: str = "gpt4",  # "mbart", "mt5", or "gpt4"
        use_reranking: bool = True,
        device: str = None
    ):
        self.translation_model_type = translation_model
        self.use_reranking = use_reranking
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load translation model if needed
        self.translator = None
        self.tokenizer = None
        if translation_model == "mbart":
            self._load_mbart()
        elif translation_model == "mt5":
            self._load_mt5()
    
    def _load_mbart(self):
        """Load mBART translation model"""
        import streamlit as st
        print("Loading mBART model (this may take 2-5 minutes on first download)...")
        if 'st' in dir():
            st.info("📥 Downloading mBART model (~2.4GB)... This only happens once!")
        
        config = TRANSLATION_MODELS["mbart"]
        self.translator = MBartForConditionalGeneration.from_pretrained(
            config["model_name"]
        ).to(self.device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(config["model_name"])
        self.src_lang = config["src_lang"]
        self.tgt_lang = config["tgt_lang"]
        print(f"✓ mBART loaded successfully on {self.device}")
    
    def _load_mt5(self):
        """Load mT5 translation model"""
        import streamlit as st
        print("Loading mT5 model (this may take 2-5 minutes on first download)...")
        if 'st' in dir():
            st.info("📥 Downloading mT5 model (~1.2GB)... This only happens once!")
        
        config = TRANSLATION_MODELS["mt5"]
        self.translator = MT5ForConditionalGeneration.from_pretrained(
            config["model_name"]
        ).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
        print(f"✓ mT5 loaded successfully on {self.device}")
    
    def translate_with_mbart(self, text: str, src_lang: str = None, tgt_lang: str = None) -> str:
        """Translate text using mBART"""
        src = src_lang or self.src_lang
        tgt = tgt_lang or self.tgt_lang
        
        self.tokenizer.src_lang = src
        encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        generated_tokens = self.translator.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    def translate_with_mt5(self, text: str, task_prefix: str = "translate English to Kannada: ") -> str:
        """Translate text using mT5"""
        input_text = task_prefix + text
        encoded = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        generated_tokens = self.translator.generate(
            **encoded,
            max_length=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=False
        )
        
        result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Clean up any remaining special tokens
        result = result.replace("<extra_id_0>", "").replace("<extra_id_1>", "").strip()
        
        # If translation failed or is empty, return original with note
        if not result or len(result) < 3:
            return f"[Translation unavailable] {text}"
        
        return result
    
    def translate_with_gpt4(self, text: str, target_lang: str = "Tulu") -> str:
        """Translate text using GPT-4 API"""
        prompt = f"Translate the following text to {target_lang}. Only provide the translation, no explanations:\n\n{text}"
        return generate_openai(prompt, max_tokens=512, temperature=0.1)
    
    def translate(self, text: str, target_lang: str = "Tulu") -> str:
        """Translate text using configured model"""
        if self.translation_model_type == "mbart":
            return self.translate_with_mbart(text)
        elif self.translation_model_type == "mt5":
            result = self.translate_with_mt5(text, task_prefix=f"translate English to Kannada: ")
            
            # If mT5 fails (produces special tokens or empty), provide helpful message
            if "<extra_id" in result or len(result.strip()) < 5:
                return f"""⚠️ mT5 translation not available for this content.

**Original English:**
{text}

💡 **Tip:** For better translation, try using mBART model instead (select in sidebar)."""
            return result
        elif self.translation_model_type == "gpt4":
            return self.translate_with_gpt4(text, target_lang)
        else:
            raise ValueError(f"Unknown translation model: {self.translation_model_type}")
    
    def retrieve_and_rank(self, query: str, top_k: int = 10, rerank_top: int = 5) -> List[Tuple[int, float, str]]:
        """
        Retrieve passages and optionally rerank them
        Returns: List of (index, score, text) tuples
        """
        # Step 1: Retrieve candidates
        raw_results = retrieve(query, top_k=top_k)
        
        if not raw_results:
            return []
        
        # Step 2: Rerank if enabled
        if self.use_reranking:
            candidates = [text for (_, _, text) in raw_results]
            ranked_indices_scores = rank(query, candidates)
            
            # Convert to format: (original_idx, rerank_score, text)
            reranked = []
            for cand_idx, score in ranked_indices_scores[:rerank_top]:
                orig_idx, _, text = raw_results[cand_idx]
                reranked.append((orig_idx, float(score), text))
            return reranked
        else:
            return raw_results[:rerank_top]
    
    def format_passages_for_prompt(self, passages: List[Tuple[int, float, str]], max_length: int = 400) -> str:
        """Format retrieved passages for inclusion in prompt"""
        formatted = []
        for i, (idx, score, text) in enumerate(passages, 1):
            # Truncate long passages
            truncated = text[:max_length].strip()
            if len(text) > max_length:
                truncated += "..."
            formatted.append(f"[Passage {i}] (relevance: {score:.3f})\n{truncated}")
        
        return "\n\n".join(formatted)
    
    def generate_answer(
        self,
        question: str,
        passages: List[Tuple[int, float, str]],
        language: str = "en",
        include_sources: bool = True
    ) -> str:
        """
        Generate answer using retrieved passages
        """
        # Format passages
        passages_text = self.format_passages_for_prompt(passages)
        
        # For non-GPT4 models, use simple extraction instead of OpenAI
        if self.translation_model_type in ["mbart", "mt5"]:
            # Simple answer from passages (no OpenAI needed)
            if not passages:
                return "No relevant passages found to answer this question."
            
            answer = f"Based on the retrieved information:\n\n"
            for i, (idx, score, text) in enumerate(passages[:3], 1):
                snippet = text[:250].strip()
                if len(text) > 250:
                    snippet += "..."
                answer += f"{snippet}\n\n"
            
            if include_sources:
                answer += "\n📚 Sources: " + ", ".join([f"Passage {i+1}" for i in range(min(3, len(passages)))])
            
            return answer
        
        # For GPT-4, use OpenAI to generate answer
        if language == "en" or language == "english":
            prompt = f"""You are an expert assistant. Use the retrieved passages below to answer the question accurately.
If the passages don't contain enough information, say so. Cite specific passages when possible.

Question: {question}

Retrieved Passages:
{passages_text}

Provide a clear, concise answer (3-5 sentences):"""
        
        elif language == "tulu" or language == "tcy":
            prompt = f"""You are a helpful assistant. Answer in Tulu using the passages below.
Do not invent information. If unsure, say "ನನಗೆ ಗೊತ್ತಿಲ್ಲ".

ಪ್ರಶ್ನೆ: {question}

ಪ್ಯಾಥ್‌ಗಳು:
{passages_text}

ಉತ್ತರ (ತುಳು):"""
        
        else:
            prompt = f"""Question: {question}

Context:
{passages_text}

Answer:"""
        
        # Generate answer
        answer = generate_openai(prompt, max_tokens=300, temperature=0.2)
        
        # Optionally add sources
        if include_sources and language == "en":
            sources = "\n\nSOURCES:\n" + "\n".join([
                f"- Passage {i+1} (ID: {idx}, Score: {score:.3f})"
                for i, (idx, score, _) in enumerate(passages)
            ])
            answer += sources
        
        return answer
    
    def query(
        self,
        question: str,
        response_language: str = "en",
        translate_question: bool = False,
        top_k: int = 10,
        rerank_top: int = 5
    ) -> Dict[str, any]:
        """
        Complete RAG query pipeline
        
        Args:
            question: User's question
            response_language: Language for response ("en", "tulu", "both")
            translate_question: Whether to translate question before retrieval
            top_k: Number of passages to retrieve
            rerank_top: Number of top passages to use after reranking
        
        Returns:
            Dictionary with answer, passages, and metadata
        """
        # Optionally translate question for better retrieval
        search_query = question
        if translate_question and response_language == "tulu":
            search_query = self.translate(question, target_lang="English")
        
        # Retrieve and rank passages
        passages = self.retrieve_and_rank(search_query, top_k=top_k, rerank_top=rerank_top)
        
        if not passages:
            return {
                "answer": "No relevant passages found.",
                "passages": [],
                "question": question,
                "language": response_language
            }
        
        # Generate answer
        if response_language == "both":
            # Generate in English first
            answer_en = self.generate_answer(question, passages, language="en")
            
            # For non-GPT4 models, translate the extracted answer
            if self.translation_model_type in ["mbart", "mt5"]:
                answer_tulu = self.translate(answer_en, target_lang="Tulu")
            else:
                # GPT4 will generate directly in Tulu
                answer_tulu = self.translate(answer_en, target_lang="Tulu")
            
            answer = f"**English:**\n{answer_en}\n\n**Tulu (Translated):**\n{answer_tulu}"
        else:
            # Generate answer in requested language
            answer_base = self.generate_answer(question, passages, language=response_language)
            
            # If Tulu requested and using mT5/mBART, translate the answer
            if response_language == "tulu" and self.translation_model_type in ["mbart", "mt5"]:
                answer = self.translate(answer_base, target_lang="Tulu")
            else:
                answer = answer_base
        
        return {
            "answer": answer,
            "passages": passages,
            "question": question,
            "search_query": search_query,
            "language": response_language,
            "num_passages": len(passages)
        }
    
    def fine_tune_translator(
        self,
        train_data_path: str,
        output_dir: str = "models/fine_tuned_translator",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5
    ):
        """
        Fine-tune translation model on domain-specific data
        
        Args:
            train_data_path: Path to JSONL file with {"source": ..., "target": ...}
            output_dir: Where to save fine-tuned model
            num_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if self.translator is None:
            raise ValueError("No translation model loaded for fine-tuning")
        
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset("json", data_files=train_data_path)
        
        # Tokenize function
        def tokenize_function(examples):
            if self.translation_model_type == "mbart":
                self.tokenizer.src_lang = self.src_lang
                model_inputs = self.tokenizer(
                    examples["source"],
                    max_length=512,
                    truncation=True,
                    padding="max_length"
                )
                
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        examples["target"],
                        max_length=512,
                        truncation=True,
                        padding="max_length"
                    )
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
            
            elif self.translation_model_type == "mt5":
                model_inputs = self.tokenizer(
                    ["translate English to Tulu: " + src for src in examples["source"]],
                    max_length=512,
                    truncation=True,
                    padding="max_length"
                )
                
                labels = self.tokenizer(
                    examples["target"],
                    max_length=512,
                    truncation=True,
                    padding="max_length"
                )
                
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs
        
        # Tokenize dataset
        tokenized = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            evaluation_strategy="no",
            warmup_steps=200,
            weight_decay=0.01,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.translator,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.translator,
            args=training_args,
            train_dataset=tokenized["train"],
            data_collator=data_collator,
        )
        
        # Train
        print(f"Starting fine-tuning for {num_epochs} epochs...")
        trainer.train()
        
        # Save
        print(f"Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("Fine-tuning complete!")


# Convenience function
def rag_answer(question: str, lang: str = "en", top_k: int = 10, rerank: bool = True) -> Tuple[str, List]:
    """
    Simplified RAG answer function (backward compatible with existing code)
    """
    pipeline = RAGPipeline(translation_model="gpt4", use_reranking=rerank)
    result = pipeline.query(question, response_language=lang, top_k=top_k)
    return result["answer"], result["passages"]


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline(translation_model="gpt4", use_reranking=True)
    
    # Test query
    result = pipeline.query(
        question="What is the history of Tulu language?",
        response_language="en",
        top_k=10,
        rerank_top=5
    )
    
    print("Question:", result["question"])
    print("\nAnswer:", result["answer"])
    print("\nTop passages:")
    for i, (idx, score, text) in enumerate(result["passages"][:3], 1):
        print(f"\n{i}. [ID: {idx}, Score: {score:.3f}]")
        print(text[:200] + "...")
