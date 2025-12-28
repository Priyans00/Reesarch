# Answer generation module using Qwen language model

from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    GENERATOR_MODEL,
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_TEMPERATURE,
    GENERATOR_TOP_P,
    SYSTEM_PROMPT,
    QA_PROMPT_TEMPLATE,
    DEVICE
)
from src.chunker import TextChunk

class Generator:
    
    # Initializes the class with configuration parameters
    def __init__(
        self,
        model_name: str = GENERATOR_MODEL,
        max_new_tokens: int = GENERATOR_MAX_NEW_TOKENS,
        temperature: float = GENERATOR_TEMPERATURE,
        top_p: float = GENERATOR_TOP_P,
        device: str = DEVICE
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        
        self.model = None
        self.tokenizer = None
    
    # Loads the pre-trained model into memory
    def load_model(self):
        if self.model is None:
            print(f"Loading generator model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
            
            print(f"  ✓ Model loaded on {self.device}")
            print(f"  ✓ Model size: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
    
    # Generates output based on input
    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT
    ) -> str:
        self.load_model()
        
        user_message = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    # Generates output based on input
    def generate_with_sources(
        self,
        question: str,
        context: str,
        source_chunks: List[TextChunk]
    ) -> Dict:
        answer = self.generate_answer(question, context)
        
        sources = []
        for i, chunk in enumerate(source_chunks, 1):
            source_info = {
                "source_id": i,
                "doc_id": chunk.doc_id,
                "section": chunk.metadata.get("section", "unknown"),
                "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "num_sources": len(sources)
        }
    
    # Checks if model abstained from answering the question
    def check_abstention(self, answer: str) -> bool:
        abstention_phrases = [
            "cannot answer",
            "not enough information",
            "context does not",
            "context doesn't",
            "not mentioned",
            "no information",
            "unable to answer",
            "cannot find",
            "not provided",
            "not available in the context"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)

class StreamingGenerator(Generator):
    
    # Generates output based on input
    def generate_stream(
        self,
        question: str,
        context: str,
        system_prompt: str = SYSTEM_PROMPT
    ):
        self.load_model()
        
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        user_message = QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer
        }
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()

class AnswerRefiner:
    
    # Initializes the class with configuration parameters
    def __init__(self, generator: Generator):
        self.generator = generator
    
    # Refines and improves initial answer for clarity and accuracy
    def refine_answer(
        self,
        question: str,
        initial_answer: str,
        context: str
    ) -> str:
        refine_prompt = f"""Original question: {question}

Initial answer: {initial_answer}

Context: {context}

Please review and improve the answer if needed. Ensure it:
1. Only uses information from the context
2. Is clear and concise
3. Properly cites the context

Refined answer:"""

        refined = self.generator.generate_answer(
            refine_prompt,
            context,
            system_prompt="You are a careful editor improving answers for accuracy and clarity."
        )
        
        return refined
    
    # Verifies if answer is supported by provided context
    def verify_answer(
        self,
        answer: str,
        context: str
    ) -> Dict:
        verify_prompt = f"""Answer: {answer}

Context: {context}

Analyze if this answer is fully supported by the context.
Respond with:
- SUPPORTED: if all claims are in the context
- PARTIALLY_SUPPORTED: if some claims are in the context
- NOT_SUPPORTED: if claims are not in the context

Your analysis:"""

        verification = self.generator.generate_answer(
            verify_prompt,
            context,
            system_prompt="You are a fact-checker verifying answers against source material."
        )
        
        verification_lower = verification.lower()
        if "not_supported" in verification_lower or "not supported" in verification_lower:
            status = "NOT_SUPPORTED"
        elif "partially" in verification_lower:
            status = "PARTIALLY_SUPPORTED"
        else:
            status = "SUPPORTED"
        
        return {
            "status": status,
            "explanation": verification
        }

if __name__ == "__main__":
    generator = Generator()