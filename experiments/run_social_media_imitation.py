import fire
import os
from automatic_prompt_engineer import ape, config, social_media_data

def run(tweet_type="original", num_prompts=50, eval_rounds=20, num_samples=50):
    """
    Runs the APE experiment for social media user imitation.

    Args:
        tweet_type (str): Type of tweets to generate/evaluate ("original" or "reply").
        num_prompts (int): Number of prompts to generate during the search.
        eval_rounds (int): Number of evaluation rounds to run.
        num_samples (int): Number of samples to use for evaluation per prompt.
    """
    print(f"Loading social media data for {tweet_type} tweets...")
    (history_inputs, history_outputs), (holdout_inputs, holdout_outputs) = \
        social_media_data.load_social_media_data(tweet_type=tweet_type)

    if not history_inputs or not holdout_inputs:
        print("Not enough data to run the experiment. Please check your data files.")
        return

    history_data = (history_inputs, history_outputs)
    holdout_data = (holdout_inputs, holdout_outputs)

    # Define templates for social media imitation
    if tweet_type == "original":
        eval_template = "User persona: [INPUT]\nGenerate an original tweet: [PROMPT]\nTweet: [OUTPUT]"
        demos_template = "User persona: [INPUT]\nGenerate an original tweet: [OUTPUT]"
        prompt_gen_template = "I gave a friend a user persona. Based on the persona they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    elif tweet_type == "reply":
        eval_template = "Generate a reply tweet: [PROMPT]\n[INPUT]\nReply: [OUTPUT]"
        demos_template = "[INPUT]\nReply: [OUTPUT]"
        prompt_gen_template = "I gave a friend a conversation. Based on this they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
    else:
        raise ValueError("Invalid tweet_type. Must be 'original' or 'reply'.")

    base_config_path = 'automatic_prompt_engineer/automatic_prompt_engineer/configs/default.yaml'
    conf = {
        'generation': {
            'num_subsamples': 5,
            'num_demos': 5,
            'num_prompts_per_subsample': num_prompts // 5, # Distribute prompts across subsamples
            'model': {
                'gpt_config': {
                    'model': 'gpt-4o'
                }
            }
        },
        'evaluation': {
            'method': 'text_similarity', # Use the new text similarity evaluator
            'num_samples': min(num_samples, len(holdout_data[0])),
            'rounds': eval_rounds,
            'model': {
                'gpt_config': {
                    'model': 'gpt-4o'
                }
            }
        }
    }

    print('Finding prompts...')
    res, demo_fn = ape.find_prompts(eval_template=eval_template,
                                    prompt_gen_data=history_data,
                                    eval_data=holdout_data,
                                    conf=conf,
                                    base_conf=base_config_path,
                                    few_shot_data=history_data, # Use history data for few-shot demos
                                    demos_template=demos_template,
                                    prompt_gen_template=prompt_gen_template)

    print('Finished finding prompts.')
    sorted_prompts_scores = res.sorted()
    print('Top Prompts:')
    for prompt, score in sorted_prompts_scores[:10]:
        print(f'  {score}: {prompt}')

    # Save results
    results_dir = f'results/social_media_imitation/{tweet_type}'
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'best_prompts.txt'), 'w') as f:
        for prompt, score in sorted_prompts_scores:
            f.write(f'{score}: {prompt}\n')
    
    print(f"Results saved to {results_dir}/best_prompts.txt")

if __name__ == '__main__':
    fire.Fire(run)