"""
Evaluation metrics and testing for the MAT chatbot
"""
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from typing import Dict, List, Tuple
from .rag_system import RAGSystem
from .logger import setup_logger
import json
import time

logger = setup_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class ChatbotEvaluator:
    """Evaluation system for the MAT chatbot"""
    
    def __init__(self):
        self.rag_system = RAGSystem()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        logger.info("ChatbotEvaluator initialized")
    
    def create_test_dataset(self) -> List[Dict]:
        """Create a test dataset with questions and expected answers"""
        test_cases = [
            {
                "question": "What are the MAT standards?",
                "expected_answer": "MAT standards are Medication-Assisted Treatment standards that provide guidelines for treatment implementation.",
                "category": "general"
            },
            {
                "question": "What is MAT Standard 1?",
                "expected_answer": "MAT Standard 1 focuses on access and choice in medication-assisted treatment.",
                "category": "specific_standard"
            },
            {
                "question": "How are MAT standards implemented?",
                "expected_answer": "MAT standards are implemented through healthcare organizations and supported by various agencies.",
                "category": "implementation"
            },
            {
                "question": "What is the aim of MAT?",
                "expected_answer": "The aim of MAT is to provide effective medication-assisted treatment for substance use disorders.",
                "category": "purpose"
            },
            {
                "question": "Which organizations support MAT?",
                "expected_answer": "Organizations like Public Health Scotland and NHS support MAT implementation.",
                "category": "organizations"
            }
        ]
        
        logger.info(f"Created test dataset with {len(test_cases)} cases")
        return test_cases
    
    def evaluate_response(self, reference: str, generated: str) -> Dict:
        """Evaluate a single response using multiple metrics"""
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, generated)
        
        # BLEU score
        reference_tokens = reference.split()
        generated_tokens = generated.split()
        bleu_score = sentence_bleu(
            [reference_tokens], 
            generated_tokens, 
            smoothing_function=self.smoothing
        )
        
        # Length-based metrics
        length_ratio = len(generated_tokens) / max(len(reference_tokens), 1)
        
        # Keyword overlap
        ref_words = set(reference.lower().split())
        gen_words = set(generated.lower().split())
        keyword_overlap = len(ref_words.intersection(gen_words)) / max(len(ref_words), 1)
        
        return {
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
            'length_ratio': length_ratio,
            'keyword_overlap': keyword_overlap
        }
    
    def run_evaluation(self, test_cases: List[Dict] = None) -> Dict:
        """Run comprehensive evaluation"""
        if test_cases is None:
            test_cases = self.create_test_dataset()
        
        logger.info("Starting comprehensive evaluation")
        
        # Setup RAG system
        if not self.rag_system.is_loaded:
            self.rag_system.setup()
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Evaluating case {i+1}/{len(test_cases)}")
            
            question = test_case['question']
            expected = test_case['expected_answer']
            category = test_case['category']
            
            # Generate response and measure time
            start_time = time.time()
            result = self.rag_system.generate_response(question)
            response_time = time.time() - start_time
            total_time += response_time
            
            generated = result['response']
            
            # Evaluate response
            metrics = self.evaluate_response(expected, generated)
            
            # Add metadata
            metrics.update({
                'question': question,
                'expected': expected,
                'generated': generated,
                'category': category,
                'response_time': response_time,
                'sources_found': len(result.get('sources', [])),
                'context_used': result.get('context_used', False)
            })
            
            results.append(metrics)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        aggregate_metrics['total_evaluation_time'] = total_time
        aggregate_metrics['average_response_time'] = total_time / len(test_cases)
        
        logger.info("Evaluation completed")
        return {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics,
            'test_cases_count': len(test_cases)
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from individual results"""
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'keyword_overlap']
        
        aggregates = {}
        for metric in metrics:
            values = [r[metric] for r in results]
            aggregates[f'{metric}_mean'] = sum(values) / len(values)
            aggregates[f'{metric}_min'] = min(values)
            aggregates[f'{metric}_max'] = max(values)
        
        # Category-wise performance
        categories = set(r['category'] for r in results)
        for category in categories:
            category_results = [r for r in results if r['category'] == category]
            for metric in metrics:
                values = [r[metric] for r in category_results]
                aggregates[f'{category}_{metric}_mean'] = sum(values) / len(values)
        
        return aggregates
    
    def save_evaluation_results(self, results: Dict, filepath: str = None):
        """Save evaluation results to JSON file"""
        if filepath is None:
            filepath = f"{config.PROCESSED_DATA_PATH}/evaluation_results.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    def generate_evaluation_report(self, results: Dict) -> str:
        """Generate a human-readable evaluation report"""
        agg = results['aggregate_metrics']
        
        report = f"""
# MAT Chatbot Evaluation Report

## Summary
- **Test Cases:** {results['test_cases_count']}
- **Total Evaluation Time:** {agg['total_evaluation_time']:.2f} seconds
- **Average Response Time:** {agg['average_response_time']:.2f} seconds

## Performance Metrics (Mean Scores)
- **ROUGE-1 F-Score:** {agg['rouge1_f_mean']:.3f}
- **ROUGE-2 F-Score:** {agg['rouge2_f_mean']:.3f}
- **ROUGE-L F-Score:** {agg['rougeL_f_mean']:.3f}
- **BLEU Score:** {agg['bleu_mean']:.3f}
- **Keyword Overlap:** {agg['keyword_overlap_mean']:.3f}

## Performance Range
- **Best ROUGE-1:** {agg['rouge1_f_max']:.3f}
- **Worst ROUGE-1:** {agg['rouge1_f_min']:.3f}
- **Best BLEU:** {agg['bleu_max']:.3f}
- **Worst BLEU:** {agg['bleu_min']:.3f}

## Category Performance
"""
        
        # Add category-wise performance
        categories = set()
        for key in agg.keys():
            if '_rouge1_f_mean' in key:
                category = key.replace('_rouge1_f_mean', '')
                categories.add(category)
        
        for category in sorted(categories):
            if f'{category}_rouge1_f_mean' in agg:
                report += f"- **{category.title()}:** ROUGE-1: {agg[f'{category}_rouge1_f_mean']:.3f}, "
                report += f"BLEU: {agg[f'{category}_bleu_mean']:.3f}\\n"
        
        return report

def run_evaluation_suite():
    """Standalone function to run evaluation"""
    evaluator = ChatbotEvaluator()
    results = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_evaluation_results(results)
    
    # Generate and print report
    report = evaluator.generate_evaluation_report(results)
    print(report)
    
    return results

if __name__ == "__main__":
    run_evaluation_suite()