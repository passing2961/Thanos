def format_messages(system_prompt, user_prompt):
    """
    Formats the system and user prompts into a list of message dictionaries.

    Parameters:
    - system_prompt (str): The prompt from the system.
    - user_prompt (str): The prompt from the user.

    Returns:
    - List[Dict[str, str]]: A list of messages formatted for dialogue processing.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def generate_incremental_sub_lists(input_list):
    """
    Generates a list of sub-lists where each sub-list incrementally expands by 2 elements.

    Parameters:
    - input_list (List): The original list to be broken down into sub-lists.

    Returns:
    - List[List]: A list of sub-lists, with each sub-list containing an increasing number of elements from the original list.
    """
    sub_lists = []
    
    for i in range(2, len(input_list) + 1, 2):
        sub_lists.append(input_list[:i])
    
    # Ensure that the last sub-list contains all elements if the total number is odd
    # if len(input_list) % 2 != 0:
    #     sub_lists.append(input_list)
    
    return sub_lists
