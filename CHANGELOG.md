# Changelog for Automatic Prompt Engineer (APE) Repository

This document details the modifications made to the `automatic_prompt_engineer` repository to update the OpenAI library, change the default model, adapt templates for social media user imitation, and integrate ROUGE/BLEU metrics for tweet quality evaluation.

## Summary of Changes

### 1. OpenAI Library Update and Model Configuration

*   **`automatic_prompt_engineer/automatic_prompt_engineer/llm.py`**:
    *   **OpenAI Client Migration**: Replaced the old `import openai` with `from openai import OpenAI` and initialized a new `OpenAI()` client instance (`self.client`) within the `GPT_Forward` and `GPT_Insert` classes.
    *   **API Call Migration**: All instances of `openai.Completion.create` were updated to `self.client.chat.completions.create`. This involved reformatting the `prompt` parameter into a `messages` list (e.g., `messages=[{"role": "user", "content": p} for p in prompt]`).
    *   **Cost Estimation Update**: The `gpt_costs_per_thousand` dictionary was expanded to include `gpt-4o` and `gpt-4o-mini` with separate input and output token costs. The `gpt_get_estimated_cost` function was refactored to correctly calculate costs based on these new input/output token prices for chat models, while retaining compatibility with older models.
    *   **Logprobs Functionality**: The `__log_probs` method and related `logprobs` handling were commented out and replaced with a `NotImplementedError`, as the new OpenAI chat completion API does not support `logprobs` in the same manner.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/configs/default.yaml`**:
    *   **Default Model Change**: The `model` fields under `generation`, `evaluation`, and `demo` sections were updated from `text-davinci-002` to `gpt-4o` to reflect the new preferred model.

### 2. Social Media User Imitation Adaptation & Data Integration

*   **`automatic_prompt_engineer/automatic_prompt_engineer/social_media_data.py` (New File)**:
    *   **Data Loading**: A new module was created to handle the loading of social media tweet data from JSONL files located in `Preprocessing/output_directory/users/`.
    *   **History and Holdout Splits**: The `load_social_media_data` function now explicitly separates and returns data for "history" and "holdout" sets, aligning with typical training and evaluation workflows.
    *   **User Persona Synthesis**: Basic user persona information (User ID, total tweets, original tweets, replies) is extracted from `user_stats.json` and incorporated into the input prompt.
    *   **Reply Tweet Input Formatting**: For "reply" tweet types, the user persona and the `previous_message` from the tweet data are combined into a single input string to fit the APE framework's `(input, output)` pair expectation.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/data.py`**:
    *   **Social Media Data Loader Integration**: The `load_social_media_data` function was imported, and a new `load_data` function was added to serve as a dispatcher, allowing the system to load "social_media" datasets using the new custom loader.
    *   **Robust Data Subsampling/Splitting**: Minor adjustments were made to `subsample_data` and `create_split` functions to ensure that the requested `subsample_size` or `split_size` does not exceed the available data length, preventing potential errors.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/template.py`**:
    *   **Flexible DemosTemplate**: The `DemosTemplate.fill` method was enhanced to dynamically handle different data formats. It now supports both `(inputs, outputs)` tuples (for original tweets) and `(inputs, previous_messages, outputs)` tuples (for reply tweets), correctly replacing the `[PREVIOUS_MESSAGE]` placeholder when present.
*   **`experiments/run_social_media_imitation.py` (New File)**:
    *   **Dedicated Experiment Script**: A new script was created to specifically run APE experiments for social media user imitation.
    *   **Data Loading**: It utilizes `social_media_data.load_social_media_data` to get the history and holdout datasets.
    *   **Social Media Specific Templates**: Defines `eval_template`, `demos_template`, and `prompt_gen_template` tailored for generating original tweets or replies, incorporating user persona and conversation history as needed.
    *   **APE Workflow Integration**: Calls `ape.find_prompts` and `ape.evaluate_prompts` with the appropriate data splits and configurations, including the `gpt-4o` model and the new `text_similarity` evaluation method.
    *   **Results Saving**: Automatically creates a directory and saves the top generated prompts and their scores to a text file.

### 3. ROUGE/BLEU Metrics Integration for Tweet Quality Evaluation

*   **`automatic_prompt_engineer/automatic_prompt_engineer/evaluation/text_similarity.py` (New File)**:
    *   **Metric Calculation Functions**: Introduced `calculate_bleu` (using `nltk`) and `calculate_rouge` (using `rouge_score`) functions to compute text similarity scores between generated and reference tweets.
    *   **NLTK Data Download**: Includes a check and automatic download for the `punkt` tokenizer if not already present.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/evaluation/text_similarity_evaluator.py` (New File)**:
    *   **Text Similarity Evaluator**: Implements `text_similarity_evaluator`, a new evaluation method that leverages the `calculate_bleu` and `calculate_rouge` functions to score generated tweets against reference tweets.
    *   **EvaluationResult Implementation**: Defines `TextSimilarityEvaluationResult` to store and sort prompts based on their aggregated BLEU and ROUGE scores.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/evaluate.py`**:
    *   **New Evaluation Method Registration**: The `get_eval_method` function was updated to recognize and return the `text_similarity_evaluator` when `'text_similarity'` is specified as the evaluation method.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/configs/default.yaml`**:
    *   **Default Evaluation Method**: The `evaluation.method` field was changed from `likelihood` to `text_similarity`, making ROUGE/BLEU the default evaluation metric for APE.

### 4. Dependency Updates

*   **`automatic_prompt_engineer/setup.py`**:
    *   **New Dependencies**: `nltk` and `rouge-score` were added to the `install_requires` list to support the new text similarity evaluation capabilities.

These changes collectively enable the `automatic_prompt_engineer` repository to perform prompt engineering experiments for social media user imitation, leveraging updated OpenAI models and advanced text similarity metrics for evaluation.

### 5. Bug Fixes and Enhancements during Testing

During the testing phase, several issues were identified and resolved to ensure the stability and functionality of the new features:

*   **`automatic_prompt_engineer/automatic_prompt_engineer/llm.py`**:
    *   **Indentation Error Fix**: Corrected an `IndentationError` in the `GPT_Insert` class definition, which was caused by an extra blank line leading to incorrect indentation.
*   **Circular Import Resolution**:
    *   The `EvaluationResult` abstract base class was moved from [`automatic_prompt_engineer/automatic_prompt_engineer/evaluate.py`](automatic_prompt_engineer/automatic_prompt_engineer/evaluate.py) to a new dedicated module: [`automatic_prompt_engineer/automatic_prompt_engineer/evaluation/base_evaluator.py`](automatic_prompt_engineer/automatic_prompt_engineer/evaluation/base_evaluator.py).
    *   Import statements in both [`automatic_prompt_engineer/automatic_prompt_engineer/evaluate.py`](automatic_prompt_engineer/automatic_prompt_engineer/evaluate.py) and [`automatic_prompt_engineer/automatic_prompt_engineer/evaluation/text_similarity_evaluator.py`](automatic_prompt_engineer/automatic_prompt_engineer/evaluation/text_similarity_evaluator.py) were updated to import `EvaluationResult` from its new location, resolving a circular import dependency.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/social_media_data.py`**:
    *   **Robust JSON Loading**: Implemented a `try-except` block around `json.loads` to gracefully handle `json.decoder.JSONDecodeError` when processing malformed lines in `.jsonl` data files. Malformed lines are now skipped, allowing the data loading process to continue without crashing.
*   **`automatic_prompt_engineer/automatic_prompt_engineer/config.py`**:
    *   **Configuration Path Resolution**: Adjusted the logic for resolving paths to configuration files (e.g., `default.yaml`, `bandits.yaml`) to use absolute paths constructed from the project root (`os.getcwd()`). This resolved `FileNotFoundError` issues caused by ambiguous relative pathing within the package structure.
*   **`experiments/run_social_media_imitation.py`**:
    *   **Evaluation Result Unpacking Fix**: Corrected a `ValueError: too many values to unpack` by modifying how the results from `res.sorted()` were handled. The `sorted()` method returns a list of `(prompt, score)` tuples, which is now correctly iterated over.