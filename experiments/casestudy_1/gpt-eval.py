import csv
import random

def create_gpt_eval_prompt(sample_row):
    """
    Constructs a GPT-based evaluation prompt using
    the row fields from the CSV.
    """
    # Extract fields from CSV row
    example_id   = sample_row["example_id"]
    prompt_len   = sample_row["prompt_len"]
    gen_text     = sample_row["gen_text"]
    block_size   = sample_row["block_size"]
    power_limit  = sample_row["power_limit"]
    
    # Potentially you have prompt text or partial-sentence 
    # in 'prompt_len' or a separate field for the original prompt. 
    # If your CSV has a separate original prompt field, 
    # parse that as well. For now, we'll pseudo-construct it:
    
    # We'll assume 'prompt_len' indicates some truncated version 
    # of the text, so for demonstration:
    prompt_text = f"(Truncated/Partial prompt length: {prompt_len} tokens)\n"
    
    # Format the text for GPT-based evaluation:
    # This is a simple example; you can expand with 
    # grammar, consistency, etc.
    eval_prompt = f"""
You are an evaluator analyzing the following model-generated text.

--- Prompt Context ---
{prompt_text}

--- Model Generation (ID: {example_id}) ---
{gen_text}

--- Additional Info ---
Block Size: {block_size}
Power Limit: {power_limit} W

Please rate:
1. Grammar (1-10)
2. Consistency with the prompt context (1-10)
3. Overall coherence (1-10)
4. Whether the generation deviates meaningfully from a baseline

Return answers in the following JSON format:

{{
  "grammar": 0,
  "consistency": 0,
  "coherence": 0,
  "deviates_baseline": false
}}
    """.strip()
    
    return eval_prompt

def main(csv_path, sample_count=10):
    """
    1. Reads the CSV from `csv_path`.
    2. Randomly picks sample_count lines.
    3. Builds GPT evaluation prompts for each sample.
    4. Prints them out (or adapt to your needs).
    """
    # Read all rows from CSV
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Safety check: if sample_count is bigger than total 
    # rows, reduce it
    if sample_count > len(rows):
        sample_count = len(rows)
    
    # Randomly sample
    sampled_rows = random.sample(rows, sample_count)
    
    # Construct and print prompts
    for i, row in enumerate(sampled_rows, 1):
        prompt_str = create_gpt_eval_prompt(row)
        print(f"\n=== Prompt #{i} ===\n{prompt_str}\n")

if __name__ == "__main__":
    # Example usage
    csv_file_path = "final_results.csv"
    main(csv_file_path, sample_count=10)
