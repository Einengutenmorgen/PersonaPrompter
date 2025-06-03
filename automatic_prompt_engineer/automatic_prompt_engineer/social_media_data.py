import json
import os
import random

def load_social_media_data(data_dir="Preprocessing/output_directory/users", user_stats_path="Preprocessing/output_directory/user_stats.json", tweet_type="original"):
    """
    Loads social media data from JSONL files and user stats,
    and formats it into (input, output) pairs for history and holdout sets.

    Args:
        data_dir (str): Directory containing user tweet JSONL files.
        user_stats_path (str): Path to the user_stats.json file.
        tweet_type (str): "original" for original tweets, "reply" for replies, or "all" for both.

    Returns:
        tuple: A tuple containing two tuples:
               ((history_inputs, history_outputs)),
               ((holdout_inputs, holdout_outputs)).
    """
    history_inputs = []
    history_outputs = []
    holdout_inputs = []
    holdout_outputs = []

    user_stats = {}
    if os.path.exists(user_stats_path):
        with open(user_stats_path, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
            for user in stats_data:
                user_stats[user['user_id']] = user

    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            user_id = filename.replace(".jsonl", "")
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"Processing file: {file_path}")
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            print(f"Skipping malformed JSON line in {file_path}: {e} - Line: {line.strip()[:100]}...")
                            continue
                        
                        # Process history tweets
                        for tweet in data.get("tweets", []):
                            if data.get("set") == "history":
                                if tweet_type == "original" and tweet.get("reply_to_id") is not None:
                                    continue
                                if tweet_type == "reply" and tweet.get("reply_to_id") is None:
                                    continue
                                
                                full_text = tweet.get("full_text")
                                if not full_text:
                                    continue

                                persona = f"User ID: {user_id}"
                                if user_id in user_stats:
                                    stats = user_stats[user_id]
                                    persona += f", Total Tweets: {stats['total_tweets_by_user']}"
                                    persona += f", Original Tweets: {stats['original_tweets_by_user']}"
                                    persona += f", Replies: {stats['replies_by_user']}"
                                
                                if tweet_type == "reply" and tweet.get("previous_message"):
                                    history_inputs.append(f"User persona: {persona}\nConversation: {tweet['previous_message']}")
                                    history_outputs.append(full_text)
                                elif tweet_type == "original" or tweet_type == "all":
                                    history_inputs.append(f"User persona: {persona}\nGenerate an original tweet:")
                                    history_outputs.append(full_text)
                            
                            # Process holdout tweets
                            elif data.get("set") == "holdout":
                                if tweet_type == "original" and tweet.get("reply_to_id") is not None:
                                    continue
                                if tweet_type == "reply" and tweet.get("reply_to_id") is None:
                                    continue
                                
                                full_text = tweet.get("full_text")
                                if not full_text:
                                    continue

                                persona = f"User ID: {user_id}"
                                if user_id in user_stats:
                                    stats = user_stats[user_id]
                                    persona += f", Total Tweets: {stats['total_tweets_by_user']}"
                                    persona += f", Original Tweets: {stats['original_tweets_by_user']}"
                                    persona += f", Replies: {stats['replies_by_user']}"
                                
                                if tweet_type == "reply" and tweet.get("previous_message"):
                                    holdout_inputs.append(f"User persona: {persona}\nConversation: {tweet['previous_message']}")
                                    holdout_outputs.append(full_text)
                                elif tweet_type == "original" or tweet_type == "all":
                                    holdout_inputs.append(f"User persona: {persona}\nGenerate an original tweet:")
                                    holdout_outputs.append(full_text)
    
    return (history_inputs, history_outputs), (holdout_inputs, holdout_outputs)

if __name__ == '__main__':
    # Example usage:
    # Example usage for original tweets
    (history_data_orig, holdout_data_orig) = load_social_media_data(tweet_type="original")
    print(f"Loaded {len(history_data_orig[0])} original tweet examples for history.")
    print(f"Loaded {len(holdout_data_orig[0])} original tweet examples for holdout.")

    if history_data_orig[0]:
        print("\nFirst history original tweet example:")
        print(f"Input: {history_data_orig[0][0]}")
        print(f"Output: {history_data_orig[1][0]}")

    if holdout_data_orig[0]:
        print("\nFirst holdout original tweet example:")
        print(f"Input: {holdout_data_orig[0][0]}")
        print(f"Output: {holdout_data_orig[1][0]}")

    # Example usage for reply tweets
    (history_inputs_replies, history_outputs_replies), (holdout_inputs_replies, holdout_outputs_replies) = load_social_media_data(tweet_type="reply")
    print(f"\nLoaded {len(history_inputs_replies)} reply tweet examples for history.")
    print(f"Loaded {len(holdout_inputs_replies)} reply tweet examples for holdout.")

    if history_inputs_replies:
        print("\nFirst history reply tweet example:")
        print(f"Input: {history_inputs_replies[0]}")
        print(f"Output: {history_outputs_replies[0]}")

    if holdout_inputs_replies:
        print("\nFirst holdout reply tweet example:")
        print(f"Input: {holdout_inputs_replies[0]}")
        print(f"Output: {holdout_outputs_replies[0]}")