# Evaluation module for assessing retrieval and answer generation quality

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import json
from pathlib import Path

from config import EVAL_METRICS
from src.chunker import TextChunk

@dataclass
class EvaluationResult:
    question: str
    answer: str
    retrieved_chunks: List[TextChunk]
    metrics: Dict[str, float] = field(default_factory=dict)
    ground_truth: Optional[str] = None
    relevant_doc_ids: List[str] = field(default_factory=list)
    
    # Converts object to dictionary representation
    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_chunks": [c.chunk_id for c in self.retrieved_chunks],
            "metrics": self.metrics,
            "ground_truth": self.ground_truth,
            "relevant_doc_ids": self.relevant_doc_ids
        }

@dataclass
class EvaluationReport:
    total_queries: int
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Converts object to dictionary representation
    def to_dict(self) -> Dict:
        return {
            "total_queries": self.total_queries,
            "aggregate_metrics": self.aggregate_metrics,
            "results": [r.to_dict() for r in self.results]
        }
    
    # Saves data to disk
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

class RetrievalEvaluator:
    
    # Initializes the class with configuration parameters
    def __init__(self):
        self.results = []
    
    # Precision At K
    def precision_at_k(
        self,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        if not retrieved_chunks or not relevant_doc_ids:
            return 0.0
        
        top_k = retrieved_chunks[:k]
        relevant_count = sum(
            1 for chunk in top_k 
            if chunk.doc_id in relevant_doc_ids
        )
        
        return relevant_count / k
    
    # Recall At K
    def recall_at_k(
        self,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        if not retrieved_chunks or not relevant_doc_ids:
            return 0.0
        
        top_k = retrieved_chunks[:k]
        retrieved_doc_ids = set(chunk.doc_id for chunk in top_k)
        relevant_retrieved = len(retrieved_doc_ids.intersection(relevant_doc_ids))
        
        return relevant_retrieved / len(relevant_doc_ids)
    
    # Mrr
    def mrr(
        self,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: List[str]
    ) -> float:
        if not retrieved_chunks or not relevant_doc_ids:
            return 0.0
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            if chunk.doc_id in relevant_doc_ids:
                return 1.0 / i
        
        return 0.0
    
    # F1 At K
    def f1_at_k(
        self,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        precision = self.precision_at_k(retrieved_chunks, relevant_doc_ids, k)
        recall = self.recall_at_k(retrieved_chunks, relevant_doc_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    # Evaluates performance using metrics
    def evaluate_retrieval(
        self,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        metrics = {}
        
        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved_chunks, relevant_doc_ids, k)
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved_chunks, relevant_doc_ids, k)
            metrics[f"f1@{k}"] = self.f1_at_k(retrieved_chunks, relevant_doc_ids, k)
        
        metrics["mrr"] = self.mrr(retrieved_chunks, relevant_doc_ids)
        
        return metrics

class AnswerEvaluator:
    
    # Initializes the class with configuration parameters
    def __init__(self):
        pass
    
    # Compute Token Overlap
    def compute_token_overlap(
        self,
        answer: str,
        context: str
    ) -> float:
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                    'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'under', 'again', 'further', 'then', 'once',
                    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                    'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                    'very', 'just', 'also', 'now', 'this', 'that', 'these',
                    'those', 'it', 'its'}
        
        answer_tokens -= stopwords
        context_tokens -= stopwords
        
        if not answer_tokens:
            return 0.0
        
        overlap = len(answer_tokens.intersection(context_tokens))
        return overlap / len(answer_tokens)
    
    # Check Hallucination Indicators
    def check_hallucination_indicators(self, answer: str) -> Dict[str, bool]:
        indicators = {
            "makes_specific_claims": False,
            "uses_hedging": False,
            "mentions_uncertainty": False,
            "references_context": False
        }
        
        answer_lower = answer.lower()
        
        specific_patterns = ['exactly', 'precisely', 'specifically', '%', 'percent']
        indicators["makes_specific_claims"] = any(p in answer_lower for p in specific_patterns)
        
        hedging_patterns = ['might', 'could', 'possibly', 'perhaps', 'seems', 'appears']
        indicators["uses_hedging"] = any(p in answer_lower for p in hedging_patterns)
        
        uncertainty_patterns = ['unclear', 'uncertain', 'not sure', 'difficult to say']
        indicators["mentions_uncertainty"] = any(p in answer_lower for p in uncertainty_patterns)
        
        context_patterns = ['according to', 'the context', 'the paper', 'source', 'states that']
        indicators["references_context"] = any(p in answer_lower for p in context_patterns)
        
        return indicators
    
    # Evaluates performance using metrics
    def evaluate_answer(
        self,
        answer: str,
        context: str,
        question: str
    ) -> Dict[str, float]:
        metrics = {}
        
        metrics["context_overlap"] = self.compute_token_overlap(answer, context)
        
        metrics["answer_length"] = min(len(answer.split()) / 100, 1.0)
        
        indicators = self.check_hallucination_indicators(answer)
        metrics["references_context"] = 1.0 if indicators["references_context"] else 0.0
        metrics["uses_hedging"] = 1.0 if indicators["uses_hedging"] else 0.0
        
        metrics["groundedness"] = (
            metrics["context_overlap"] * 0.6 +
            metrics["references_context"] * 0.3 +
            (1 - metrics["uses_hedging"]) * 0.1
        )
        
        return metrics

class RAGEvaluator:
    
    # Initializes the class with configuration parameters
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerEvaluator()
        self.results: List[EvaluationResult] = []
    
    # Evaluates performance using metrics
    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str,
        retrieved_chunks: List[TextChunk],
        relevant_doc_ids: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        metrics = {}
        
        if relevant_doc_ids:
            retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
                retrieved_chunks, relevant_doc_ids
            )
            metrics.update(retrieval_metrics)
        
        answer_metrics = self.answer_evaluator.evaluate_answer(answer, context, question)
        metrics.update(answer_metrics)
        
        result = EvaluationResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            metrics=metrics,
            ground_truth=ground_truth,
            relevant_doc_ids=relevant_doc_ids or []
        )
        
        self.results.append(result)
        return result
    
    # Evaluates performance using metrics
    def evaluate_batch(
        self,
        qa_pairs: List[Dict]
    ) -> EvaluationReport:
        for pair in qa_pairs:
            self.evaluate_single(
                question=pair["question"],
                answer=pair["answer"],
                context=pair.get("context", ""),
                retrieved_chunks=pair.get("retrieved_chunks", []),
                relevant_doc_ids=pair.get("relevant_doc_ids"),
                ground_truth=pair.get("ground_truth")
            )
        
        return self.generate_report()
    
    # Generates output based on input
    def generate_report(self) -> EvaluationReport:
        if not self.results:
            return EvaluationReport(total_queries=0, results=[], aggregate_metrics={})
        
        all_metrics = defaultdict(list)
        for result in self.results:
            for metric_name, value in result.metrics.items():
                all_metrics[metric_name].append(value)
        
        aggregate_metrics = {
            name: np.mean(values) for name, values in all_metrics.items()
        }
        
        return EvaluationReport(
            total_queries=len(self.results),
            results=self.results,
            aggregate_metrics=aggregate_metrics
        )
    
    # Reset
    def reset(self):
        self.results = []
    
    # Print Summary
    def print_summary(self, report: EvaluationReport):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total queries evaluated: {report.total_queries}")
        print("\nAggregate Metrics:")
        
        for metric, value in sorted(report.aggregate_metrics.items()):
            print(f"  {metric}: {value:.4f}")
        
        print("="*60)

class FailureCaseAnalyzer:
    
    # Initializes the class with configuration parameters
    def __init__(self, evaluator: RAGEvaluator):
        self.evaluator = evaluator
    
    # Identify Failures
    def identify_failures(
        self,
        threshold: float = 0.3
    ) -> Dict[str, List[EvaluationResult]]:
        failures = {
            "retrieval_failure": [],
            "low_groundedness": [],
            "abstention": [],
            "short_answer": []
        }
        
        for result in self.evaluator.results:
            if result.relevant_doc_ids:
                mrr = result.metrics.get("mrr", 0)
                if mrr < threshold:
                    failures["retrieval_failure"].append(result)
            
            groundedness = result.metrics.get("groundedness", 0)
            if groundedness < threshold:
                failures["low_groundedness"].append(result)
            
            if len(result.answer.split()) < 10:
                failures["short_answer"].append(result)
        
        return failures
    
    # Analyze Patterns
    def analyze_patterns(self, failures: Dict[str, List[EvaluationResult]]) -> Dict:
        analysis = {}
        
        for failure_type, cases in failures.items():
            if not cases:
                continue
            
            analysis[failure_type] = {
                "count": len(cases),
                "percentage": len(cases) / len(self.evaluator.results) * 100,
                "example_questions": [c.question for c in cases[:3]]
            }
        
        return analysis
    
    # Generates output based on input
    def generate_failure_report(self) -> Dict:
        failures = self.identify_failures()
        patterns = self.analyze_patterns(failures)
        
        return {
            "total_evaluated": len(self.evaluator.results),
            "failure_patterns": patterns,
            "recommendations": self._generate_recommendations(patterns)
        }
    
    # Generates output based on input
    def _generate_recommendations(self, patterns: Dict) -> List[str]:
        recommendations = []
        
        if "retrieval_failure" in patterns and patterns["retrieval_failure"]["percentage"] > 20:
            recommendations.append(
                "High retrieval failure rate. Consider: "
                "1) Improving chunking strategy "
                "2) Using hybrid retrieval (dense + sparse) "
                "3) Adding query expansion"
            )
        
        if "low_groundedness" in patterns and patterns["low_groundedness"]["percentage"] > 30:
            recommendations.append(
                "Low answer groundedness. Consider: "
                "1) Using stricter prompts "
                "2) Reducing temperature "
                "3) Adding verification step"
            )
        
        if "short_answer" in patterns and patterns["short_answer"]["percentage"] > 40:
            recommendations.append(
                "Many short answers. This might indicate: "
                "1) Insufficient context being retrieved "
                "2) Model abstaining too often "
                "3) Questions outside document scope"
            )
        
        return recommendations

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    test_chunks = [
        TextChunk(chunk_id="c1", doc_id="doc1", text="Test chunk 1"),
        TextChunk(chunk_id="c2", doc_id="doc2", text="Test chunk 2"),
    ]
    
    result = evaluator.evaluate_single(
        question="What is the main contribution?",
        answer="According to the context, the main contribution is a novel approach.",
        context="The paper presents a novel approach for text processing.",
        retrieved_chunks=test_chunks,
        relevant_doc_ids=["doc1"]
    )
    
    print("Evaluation result:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    report = evaluator.generate_report()
    evaluator.print_summary(report)
