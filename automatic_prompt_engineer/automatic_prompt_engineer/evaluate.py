from automatic_prompt_engineer import llm
from automatic_prompt_engineer.evaluation import text_similarity_evaluator
from automatic_prompt_engineer.evaluation.base_evaluator import EvaluationResult



def get_eval_method(eval_method):
    """
    Returns the evaluation method object.
    Parameters:
        eval_method: The evaluation method to use. ('likelihood')
    Returns:
        An evaluation method object.
    """
    if callable(eval_method):
        return eval_method
    if eval_method == 'likelihood':
        from automatic_prompt_engineer.evaluation import likelihood
        return likelihood.likelihood_evaluator
    elif eval_method == 'bandits':
        from automatic_prompt_engineer.evaluation import bandits
        return bandits.bandits_evaluator
    elif eval_method == 'text_similarity':
        return text_similarity_evaluator.text_similarity_evaluator
    else:
        raise ValueError('Invalid evaluation method.')


def evalute_prompts(prompts, eval_template, eval_data, demos_template, few_shot_data, eval_method, config):
    """
    Returns the scores for a list of prompts.
    Parameters:
        prompts: A list of prompts.
        eval_template: The template for the evaluation queries.
        eval_data: The data to use for evaluation.
        eval_method: The evaluation method to use. ('likelihood')
        config: The configuration dictionary.
    Returns:
        An evaluation result object.
    """
    eval_method = get_eval_method(eval_method)
    return eval_method(prompts, eval_template, eval_data, demos_template, few_shot_data, config)


def demo_function(eval_template, config):
    """
    Returns a function that can be manually test the LLM with a chosen prompt.
    Parameters:
        eval_template: The template for the evaluation queries.
        config: The configuration dictionary.
    Returns:
        A function that takes a prompt and returns a demo.
    """
    model = llm.model_from_config(config['model'])

    def fn(prompt, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        queries = []
        for input_ in inputs:
            query = eval_template.fill(prompt=prompt, input=input_)
            queries.append(query)
        outputs = model.generate_text(
            queries, n=1)
        return [out.strip().split('\n')[0] for out in outputs]

    return fn


