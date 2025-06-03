import numpy as np
from tqdm import tqdm

from automatic_prompt_engineer import llm
from automatic_prompt_engineer.evaluation.base_evaluator import EvaluationResult
from automatic_prompt_engineer.evaluation.text_similarity import calculate_bleu, calculate_rouge
from automatic_prompt_engineer import data

class TextSimilarityEvaluationResult(EvaluationResult):
    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def sorted(self, method='default'):
        # Sort by the sum of BLEU and ROUGE-L scores for now
        # This can be customized to prioritize specific metrics
        if method == 'default':
            # Ensure scores are numeric before sorting
            numeric_scores = []
            for s in self.scores:
                try:
                    numeric_scores.append(s['bleu'] + s['rougeL'])
                except TypeError: # Handle cases where scores might be None or non-numeric
                    numeric_scores.append(-float('inf')) # Place non-numeric scores at the end
            sorted_indices = np.argsort([-s for s in numeric_scores])
        else:
            raise ValueError(f"Unknown sorting method: {method}")
        
        return [(self.prompts[i], self.scores[i]) for i in sorted_indices]

    def in_place(self, method='default'):
        return [(self.prompts[i], self.scores[i]) for i in range(len(self.prompts))]

def text_similarity_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    """
    Evaluates prompts based on text similarity (BLEU and ROUGE) between generated and reference outputs.
    
    Args:
        prompts (list): List of prompts to evaluate.
        eval_template: The template for the evaluation queries.
        eval_data (tuple): Tuple of (inputs, outputs) for evaluation.
        demos_template: The template for the demos.
        few_shot_data (tuple): Data for few-shot demonstrations.
        config (dict): Configuration dictionary.
        
    Returns:
        TextSimilarityEvaluationResult: An object containing prompts and their similarity scores.
    """
    model = llm.model_from_config(config['model'])
    
    all_prompt_scores = []

    for prompt in tqdm(prompts, desc="Evaluating prompts with text similarity"):
        prompt_scores = {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        # Subsample data for evaluation
        eval_inputs, eval_outputs = data.subsample_data(eval_data, config['num_samples'])
        
        queries = []
        for input_text in eval_inputs:
            # Fill the evaluation template with the current prompt and input
            query = eval_template.fill(prompt=prompt, input=input_text)
            queries.append(query)
        
        # Generate text for all queries for the current prompt
        generated_texts = model.generate_text(queries, n=1) # n=1 for evaluation
        
        for i, generated_text in enumerate(generated_texts):
            reference_text = eval_outputs[i]
            
            # Calculate BLEU
            bleu = calculate_bleu(reference_text, generated_text)
            prompt_scores['bleu'].append(bleu)
            
            # Calculate ROUGE
            rouge = calculate_rouge(reference_text, generated_text)
            prompt_scores['rouge1'].append(rouge['rouge1'])
            prompt_scores['rouge2'].append(rouge['rouge2'])
            prompt_scores['rougeL'].append(rouge['rougeL'])
        
        # Average scores for the current prompt
        avg_scores = {k: np.mean(v) for k, v in prompt_scores.items()}
        all_prompt_scores.append(avg_scores)
        
    return TextSimilarityEvaluationResult(prompts, all_prompt_scores)