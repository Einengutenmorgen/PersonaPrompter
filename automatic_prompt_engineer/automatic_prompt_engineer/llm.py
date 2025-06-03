"""Contains classes for querying large language models."""
from math import ceil
import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# Costs per thousand tokens
# gpt-4o: Input $5.00, Output $15.00 (per million tokens, so per thousand is $0.005 and $0.015)
# gpt-4o-mini: Input $0.15, Output $0.60 (per million tokens, so per thousand is $0.00015 and $0.0006)
gpt_costs_per_thousand = {
    'davinci': {'input': 0.0200, 'output': 0.0200},
    'curie': {'input': 0.0020, 'output': 0.0020},
    'babbage': {'input': 0.0005, 'output': 0.0005},
    'ada': {'input': 0.0004, 'output': 0.0004},
    'gpt-4o': {'input': 0.005, 'output': 0.015},
    'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006}
}


def model_from_config(config, disable_tqdm=True):
    """Returns a model based on the config."""
    model_type = config["name"]
    if model_type == "GPT_forward":
        return GPT_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "GPT_insert":
        return GPT_Insert(config, disable_tqdm=disable_tqdm)
    raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for large language models."""

    @abstractmethod
    def generate_text(self, prompt):
        """Generates text from the model.
        Parameters:
            prompt: The prompt to use. This can be a string or a list of strings.
        Returns:
            A list of strings.
        """
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        """Returns the log probs of the text.
        Parameters:
            text: The text to get the log probs of. This can be a string or a list of strings.
            log_prob_range: The range of characters within each string to get the log_probs of. 
                This is a list of tuples of the form (start, end).
        Returns:
            A list of log probs.
        """
        pass


class GPT_Forward(LLM):
    """Wrapper for GPT-3 and GPT-4o models."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            # For chat models, we need to estimate input and output tokens separately
            # This is a simplified estimation, actual tokenization might differ
            model_name = self.config['gpt_config']['model']
            if 'gpt-4o' in model_name or 'gpt-3.5' in model_name: # Assuming gpt-3.5 also uses chat completion
                # A rough estimate: prompt tokens are input, max_tokens are output
                input_tokens = len(text) // 4 # Rough estimate for tokens
                output_tokens = max_tokens
                total_estimated_cost += gpt_get_estimated_cost(
                    self.config, input_tokens, output_tokens) * n
            else:
                total_estimated_cost += gpt_get_estimated_cost(
                    self.config, len(text) // 4, max_tokens) * n # Old models use total tokens
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        text = []

        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            prompt = [prompt]
        batch_size = self.config['batch_size']
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, "
                f"split into {len(prompt_batches)} batches of size {batch_size * n}")
        res = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            res += self.__complete(prompt_batch, n)
        return res

    def log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text."""
        if not isinstance(text, list):
            text = [text]
        if self.needs_confirmation:
            self.confirm_cost(text, 1, 0)
        batch_size = self.config['batch_size']
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, log_prob_range in tqdm(list(zip(text_batches, log_prob_range_batches)),
                                               disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self.__log_probs(
                text_batch, log_prob_range)
            log_probs += log_probs_batch
            tokens += tokens_batch
        return log_probs, tokens

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        while response is None:
            try:
                # For chat models, we need to format the prompt as messages
                messages = [{"role": "user", "content": p} for p in prompt]
                response = self.client.chat.completions.create(
                    model=config['model'],
                    messages=messages,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 100),
                    n=config.get('n', 1),
                    stop=config.get('stop', None)
                )
            except Exception as e:
                if 'is greater than the maximum' in str(e):
                    raise BatchSizeException()
                print(e)
                print('Retrying...')
                time.sleep(5)

        return [choice.message.content for choice in response.choices]

    def __complete(self, prompt, n):
        """Generates text from the model and returns the log prob data."""
        if not isinstance(prompt, list):
            text = [prompt]
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # If there are any [APE] tokens in the prompts, remove them
        for i in range(len(prompt)):
            prompt[i] = prompt[i].replace('[APE]', '').strip()
        response = None
        while response is None:
            try:
                # For chat models, we need to format the prompt as messages
                messages = [{"role": "user", "content": p} for p in prompt]
                response = self.client.chat.completions.create(
                    model=config['model'],
                    messages=messages,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 100),
                    n=config.get('n', 1),
                    stop=config.get('stop', None)
                )
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)
        return response.choices

    def __log_probs(self, text, log_prob_range=None):
        """Returns the log probs of the text.
        Note: The new OpenAI chat completion API does not directly support logprobs in the same way as the old Completion API.
        This method is currently not implemented for chat models.
        """
        raise NotImplementedError("Logprobs are not directly supported with the new chat completion API in the same way. This functionality needs to be re-implemented or re-evaluated.")

    def get_token_indices(self, offsets, log_prob_range):
        """Returns the indices of the tokens in the log probs that correspond to the tokens in the log_prob_range.
        Note: This method is dependent on the logprobs functionality which is currently not implemented for chat models.
        """
        raise NotImplementedError("get_token_indices is dependent on logprobs functionality which is currently not implemented for chat models.")


class GPT_Insert(LLM):
    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the model."""
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        self.client = OpenAI()

    def confirm_cost(self, texts, n, max_tokens):
        total_estimated_cost = 0
        for text in texts:
            # For chat models, we need to estimate input and output tokens separately
            model_name = self.config['gpt_config']['model']
            if 'gpt-4o' in model_name or 'gpt-3.5' in model_name: # Assuming gpt-3.5 also uses chat completion
                # A rough estimate: prompt tokens are input, max_tokens are output
                input_tokens = len(text) // 4 # Rough estimate for tokens
                output_tokens = max_tokens
                total_estimated_cost += gpt_get_estimated_cost(
                    self.config, input_tokens, output_tokens) * n
            else:
                total_estimated_cost += gpt_get_estimated_cost(
                    self.config, len(text) // 4, max_tokens) * n # Old models use total tokens
        print(f"Estimated cost: ${total_estimated_cost:.2f}")
        # Ask the user to confirm in the command line
        if os.getenv("LLM_SKIP_CONFIRM") is None:
            confirm = input("Continue? (y/n) ")
            if confirm != 'y':
                raise Exception("Aborted.")

    def auto_reduce_n(self, fn, prompt, n):
        """Reduces n by half until the function succeeds."""
        try:
            return fn(prompt, n)
        except BatchSizeException as e:
            if n == 1:
                raise e
            return self.auto_reduce_n(fn, prompt, n // 2) + self.auto_reduce_n(fn, prompt, n // 2)

    def generate_text(self, prompt, n):
        if not isinstance(prompt, list):
            prompt = [prompt]
        if self.needs_confirmation:
            self.confirm_cost(
                prompt, n, self.config['gpt_config']['max_tokens'])
        batch_size = self.config['batch_size']
        assert batch_size == 1
        prompt_batches = [prompt[i:i + batch_size]
                          for i in range(0, len(prompt), batch_size)]
        if not self.disable_tqdm:
            print(
                f"[{self.config['name']}] Generating {len(prompt) * n} completions, split into {len(prompt_batches)} batches of (maximum) size {batch_size * n}")
        text = []
        for prompt_batch in tqdm(prompt_batches, disable=self.disable_tqdm):
            text += self.auto_reduce_n(self.__generate_text, prompt_batch, n)
        return text

    def log_probs(self, text, log_prob_range=None):
        raise NotImplementedError

    def __generate_text(self, prompt, n):
        """Generates text from the model."""
        config = self.config['gpt_config'].copy()
        config['n'] = n
        # Split prompts into prefixes and suffixes with the [APE] token (do not include the [APE] token in the suffix)
        prefix = prompt[0].split('[APE]')[0]
        suffix = prompt[0].split('[APE]')[1]
        response = None
        while response is None:
            try:
                # For chat models, we need to format the prompt as messages
                messages = [{"role": "user", "content": prefix + suffix}]
                response = self.client.chat.completions.create(
                    model=config['model'],
                    messages=messages,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 100),
                    n=config.get('n', 1),
                    stop=config.get('stop', None)
                )
            except Exception as e:
                print(e)
                print('Retrying...')
                time.sleep(5)

        # Remove suffix from the generated text
        texts = [choice.message.content.replace(suffix, '') for choice in response.choices]
        return texts


def gpt_get_estimated_cost(config, input_tokens, output_tokens):
    """Uses the current API costs/1000 tokens to estimate the cost of generating text from the model."""
    model_name = config['gpt_config']['model']
    
    # Determine if it's a chat model (gpt-4o, gpt-3.5 etc.) or an older completion model
    if 'gpt-4o' in model_name or 'gpt-3.5' in model_name:
        input_cost_per_thousand = gpt_costs_per_thousand[model_name]['input']
        output_cost_per_thousand = gpt_costs_per_thousand[model_name]['output']
        
        cost = (input_tokens / 1000) * input_cost_per_thousand + \
               (output_tokens / 1000) * output_cost_per_thousand
    else:
        # For older models, use a single cost per thousand tokens
        engine = model_name.split('-')[1] if '-' in model_name else model_name.split(':')[0]
        
        # Check if it's a fine-tuned model
        if engine not in gpt_costs_per_thousand:
            # Assuming fine-tuned models have different pricing
            fine_tuned_costs_per_thousand = {
                'davinci': {'input': 0.1200, 'output': 0.1200},
                'curie': {'input': 0.0120, 'output': 0.0120},
                'babbage': {'input': 0.0024, 'output': 0.0024},
                'ada': {'input': 0.0016, 'output': 0.0016}
            }
            cost_per_thousand = fine_tuned_costs_per_thousand[engine]['input'] # Assuming input/output same for fine-tuned
        else:
            cost_per_thousand = gpt_costs_per_thousand[engine]['input'] # Assuming input/output same for older models

        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * cost_per_thousand
        
    return cost


class BatchSizeException(Exception):
    pass
