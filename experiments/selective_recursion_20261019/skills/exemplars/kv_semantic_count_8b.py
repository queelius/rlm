"""Count records with specific action and item category from input data.

answer_gen_kv_semantic_count_answer_line_s1(request) -> str    # Returns the final answer text in 'ANSWER: <count>' format.
"""

import re

def answer_gen_kv_semantic_count_answer_line_s1(request) -> str:
    input_str = request['input']
    
    # Split input into data and question
    question_start = input_str.find("Question: ")
    if question_start == -1:
        return "ANSWER: 0"
    data_part = input_str[:question_start].strip()
    question_part = input_str[question_start + len("Question: "):].strip()
    
    # Parse action and category from question
    action_match = re.search(r'action (\w+) involve a (\w+) item', question_part, re.IGNORECASE)
    if not action_match:
        return "ANSWER: 0"
    desired_action = action_match.group(1).lower()
    desired_category = action_match.group(2).lower()
    
    # Process each line in data_part
    count = 0
    prompts = []
    lines = data_part.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split into key-value pairs
        parts = line.split()
        action = None
        item = None
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 'action':
                    action = value.strip().lower()
                elif key == 'item':
                    item = value.strip().lower()
        if action == desired_action and item is not None:
            # Collect prompt
            prompt = f"Is the item '{item}' considered a {desired_category} item? Answer with 'yes' or 'no'."
            prompts.append(prompt)
    
    if prompts:
        responses = ask_batch(prompts)
        for response in responses.texts:
            if response.strip().lower() == 'yes':
                count += 1
    return f"ANSWER: {count}"
