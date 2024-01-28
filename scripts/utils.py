import random

DATASET_LIST = [
    "SChem5Labels",
    "Sentiment",
    "SBIC",
    "ghc"
]

DATASET_LABELS = {
    "SChem5Labels": [0, 1, 2, 3, 4],
    "Sentiment": [0, 1, 2, 3, 4],
    "SBIC": [0, 1, 2],
    "ghc": [0, 1]
}

def calculate_majority(seed, lst): # length of list is always 5 as 5 annotators
    random.seed(seed)

    # Remove -1 values from the list
    valid_values = [x for x in lst if x != "-1"]

    # If the list is empty after removing -1 values, return None
    if not valid_values:
        return None

    # Create a dictionary to count occurrences of each value
    count_dict = {}
    for value in valid_values:
        count_dict[value] = count_dict.get(value, 0) + 1

    # Find the maximum occurrence
    max_count = max(count_dict.values())

    # If the maximum occurrence is 1, then all values are unique
    if max_count == 1:
        return random.choice(valid_values)

    # Extract all values that have the maximum occurrence
    majority_values = [key for key, value in count_dict.items() if value == max_count]

    if len(majority_values) >= 2:
        return random.choice(majority_values)
    else:
        return majority_values[0]