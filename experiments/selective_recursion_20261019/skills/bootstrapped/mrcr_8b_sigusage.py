# families: mrcr
# bootstrapped by bootstrap_library.py (proposer qwen3-8b-awq, E21; teaching mean 1.000, 30/30 exact; round 1 candidate 5)
# E25d documentation-form ablation (mrcr_8b_sigusage.py); reviewed 2026-10-22: pure string parsing, no imports, no I/O, no helper calls, nothing specific to the shown examples
"""
answer_mrcr(request) -> str
Typical use: FINAL_TEXT(answer_mrcr(request)).
"""
def answer_mrcr(request) -> str:
    messages = request['input']
    if not messages:
        return ''
    last_message = messages[-1]
    final_message_content = last_message['content']
    
    # Extract prepended string
    prepended_str = final_message_content.split('Prepend ')[1].split(' to the ')[0]
    
    # Extract n
    after_to_the = final_message_content.split('to the ')[1]
    parts = after_to_the.split()
    number_str = parts[0]
    n = int(number_str[:-2])
    
    # Extract full topic
    topic_part = final_message_content.split('to the ')[1].split('.')[0]
    topic_parts = topic_part.split()
    full_topic = ' '.join(topic_parts[3:])
    
    # Find user messages that start with 'write a ' + full_topic
    user_indices = []
    for idx, msg in enumerate(messages):
        if msg['role'] == 'user' and msg['content'].startswith('write a ' + full_topic):
            user_indices.append(idx)
    
    # Collect assistant responses to these user messages
    assistant_list = []
    for idx in user_indices:
        next_idx = idx + 1
        if next_idx < len(messages):
            next_msg = messages[next_idx]
            if next_msg['role'] == 'assistant':
                assistant_list.append(next_msg)
    
    # Select the nth assistant response
    if n <= len(assistant_list):
        target_msg = assistant_list[n-1]
        return prepended_str + target_msg['content']
    else:
        return ''
