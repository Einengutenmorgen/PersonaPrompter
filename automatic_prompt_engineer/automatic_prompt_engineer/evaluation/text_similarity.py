import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(reference, hypothesis):
    """
    Calculates the BLEU score between a reference and hypothesis sentence.
    
    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        
    Returns:
        float: The BLEU score.
    """
    # BLEU expects tokenized sentences
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    
    # Using SmoothingFunction() to avoid zero BLEU scores for short sentences
    # or when there are no common n-grams.
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothie)

def calculate_rouge(reference, hypothesis, rouge_types=['rouge1', 'rouge2', 'rougeL']):
    """
    Calculates ROUGE scores (rouge1, rouge2, rougeL) between a reference and hypothesis sentence.
    
    Args:
        reference (str): The reference sentence.
        hypothesis (str): The hypothesis sentence.
        rouge_types (list): A list of ROUGE types to calculate.
        
    Returns:
        dict: A dictionary where keys are ROUGE types and values are their f-measures.
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    
    results = {}
    for r_type in rouge_types:
        results[r_type] = scores[r_type].fmeasure
    return results

if __name__ == '__main__':
    reference_tweet = "This is a test tweet for social media imitation."
    hypothesis_tweet = "This is a test tweet for social media."
    
    bleu_score = calculate_bleu(reference_tweet, hypothesis_tweet)
    print(f"BLEU score: {bleu_score:.4f}")
    
    rouge_scores = calculate_rouge(reference_tweet, hypothesis_tweet)
    print(f"ROUGE scores: {rouge_scores}")

    reference_tweet_2 = "The quick brown fox jumps over the lazy dog."
    hypothesis_tweet_2 = "The quick brown fox jumps over the dog."
    
    bleu_score_2 = calculate_bleu(reference_tweet_2, hypothesis_tweet_2)
    print(f"\nBLEU score 2: {bleu_score_2:.4f}")
    
    rouge_scores_2 = calculate_rouge(reference_tweet_2, hypothesis_tweet_2)
    print(f"ROUGE scores 2: {rouge_scores_2}")