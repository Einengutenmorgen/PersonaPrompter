import os
import yaml


def update_config(config, base_config_relative_to_project_root='automatic_prompt_engineer\\automatic_prompt_engineer\\configs\\default.yaml'):
    # Get default config from yaml
    # Assume current working directory is the project root
    project_root = os.getcwd()
    
    config_file_path = os.path.join(project_root, base_config_relative_to_project_root)

    with open(config_file_path) as f:
        default_config = yaml.safe_load(f)

    # Update default config with user config
    # Note that the config is a nested dictionary, so we need to update it recursively
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, config)


def simple_config(eval_model, prompt_gen_model, prompt_gen_mode, num_prompts, eval_rounds, prompt_gen_batch_size, eval_batch_size):
    """Returns a config and splits the data into sensible chunks."""
    # This call needs to be updated to pass the full path from project root
    # For 'configs/bandits.yaml', the full path from project root would be
    # 'automatic_prompt_engineer/automatic_prompt_engineer/configs/bandits.yaml'
    conf = update_config({}, 'automatic_prompt_engineer/automatic_prompt_engineer/configs/bandits.yaml')
    conf['generation']['model']['gpt_config']['model'] = prompt_gen_model
    if prompt_gen_mode == 'insert':
        conf['generation']['model']['name'] = 'GPT_insert'
    elif prompt_gen_mode == 'forward':
        conf['generation']['model']['name'] = 'GPT_forward'
    conf['generation']['num_subsamples'] = num_prompts // 10
    conf['generation']['num_prompts_per_subsample'] = 10

    conf['evaluation']['base_eval_config']['model']['gpt_config']['model'] = eval_model
    conf['evaluation']['base_eval_config']['model']['batch_size'] = eval_batch_size
    # total eval = rounds * num_prompts_per_round * num_samples
    # We fix the number of samples to 10 and the number of prompts per round to 1/3 of
    # the total number of prompts. We then set the number of rounds to be the number of
    # prompts divided by the number of prompts per round.
    conf['evaluation']['num_prompts_per_round'] = 0.334
    conf['evaluation']['rounds'] = eval_rounds
    conf['evaluation']['base_eval_config']['num_samples'] = 5
    # In this simple demo, there is no dataset splitting, so we just use the same data for prompt generation and evaluation
    return conf
