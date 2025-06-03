import nltk
import json
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'), 
        ('corpora/omw-1.4', 'omw-1.4'),
        ('corpora/stopwords', 'stopwords')  # Für bessere Text-Verarbeitung
    ]
    
    for resource_path, download_name in required_data:
        try:
            nltk.data.find(resource_path)
            print(f"✓ {download_name} already available")
        except LookupError:
            print(f"Downloading {download_name}...")
            try:
                nltk.download(download_name, quiet=True)
                print(f"✓ {download_name} downloaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not download {download_name}: {e}")
    
    print("NLTK data check completed.")

def calculate_rouge_scores(reference: str, hypothesis: str) -> dict:
    """
    Calculates ROUGE scores (1, 2, L) for a given hypothesis against a reference.

    Args:
        reference (str): The reference text (ground truth).
        hypothesis (str): The hypothesis text (generated).

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L f-scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        'rouge1_fscore': scores['rouge1'].fmeasure,
        'rouge2_fscore': scores['rouge2'].fmeasure,
        'rougeL_fscore': scores['rougeL'].fmeasure,
    }

def calculate_bleu_score(reference: str, hypothesis: str) -> float:
    """
    Calculates the BLEU score for a given hypothesis against a reference.

    Args:
        reference (str): The reference text (ground truth).
        hypothesis (str): The hypothesis text (generated).

    Returns:
        float: The BLEU score.
    """
    # Tokenize sentences for BLEU calculation
    reference_tokens = nltk.word_tokenize(reference.lower())
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())

    # BLEU expects a list of reference sentences, even if there's only one
    references = [reference_tokens]

    # Use SmoothingFunction() to avoid zero BLEU scores for short sentences or no overlap
    smoothie = SmoothingFunction().method1
    return sentence_bleu(references, hypothesis_tokens, smoothing_function=smoothie)

def run_evaluation(
    imitations_input_file: Path,
    evaluation_output_file: Path
):
    """
    Loads generated imitations, calculates ROUGE and BLEU scores against original tweets,
    and saves the evaluation results.

    Args:
        imitations_input_file (Path): Path to the JSON file containing generated imitations.
        evaluation_output_file (Path): Path to save the evaluation results JSON.
    """
    print(f"Starting evaluation process...")
    download_nltk_data()

    if not imitations_input_file.exists():
        print(f"ERROR: Imitations input file not found at {imitations_input_file}")
        return

    try:
        with open(imitations_input_file, 'r', encoding='utf-8') as f:
            imitations_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load imitations data from {imitations_input_file}: {e}")
        return

    print(f"Loaded {len(imitations_data)} imitation entries for evaluation.")

    evaluation_results = []
    for i, entry in enumerate(imitations_data):
        user_id = entry.get("user_id", "N/A")
        persona_index = entry.get("persona_index", "N/A")
        imitation_strategy = entry.get("imitation_strategy", "N/A")
        holdout_tweet_id = entry.get("holdout_tweet_id", "N/A")
        
        generated_text = entry.get("generated_imitation_text", "")
        original_text = entry.get("original_holdout_tweet_text", "")

        if not generated_text or not original_text:
            print(f"WARNING: Skipping entry {i} due to missing generated or original text for user {user_id}, tweet {holdout_tweet_id}.")
            continue

        rouge_scores = calculate_rouge_scores(original_text, generated_text)
        bleu_score = calculate_bleu_score(original_text, generated_text)

        result_entry = {
            "user_id": user_id,
            "persona_index": persona_index,
            "imitation_strategy": imitation_strategy,
            "holdout_tweet_id": holdout_tweet_id,
            "original_holdout_tweet_text": original_text,
            "generated_imitation_text": generated_text,
            "rouge_scores": rouge_scores,
            "bleu_score": bleu_score
        }
        evaluation_results.append(result_entry)
    
    # Ensure output directory exists
    evaluation_output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(evaluation_output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
        print(f"Evaluation results saved to {evaluation_output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save evaluation results to {evaluation_output_file}: {e}")

# # Optional: Add a main block for testing
# if __name__ == "__main__":
#     # Test the functions
#     print("Testing evaluation_metrics module...")
#     print("Available functions:", [name for name in dir() if callable(eval(name)) and not name.startswith('_')])