import random
from automatic_prompt_engineer.social_media_data import load_social_media_data

def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), min(subsample_size, len(inputs))) # Ensure subsample_size doesn't exceed data size
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs


def create_split(data, split_size):
    """
    Split data into two parts. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    
    # Ensure split_size does not exceed the total number of samples
    actual_split_size = min(split_size, len(inputs))
    
    indices = random.sample(range(len(inputs)), actual_split_size)
    inputs1 = [inputs[i] for i in indices]
    outputs1 = [outputs[i] for i in indices]
    inputs2 = [inputs[i] for i in range(len(inputs)) if i not in indices]
    outputs2 = [outputs[i] for i in range(len(inputs)) if i not in indices]
    return (inputs1, outputs1), (inputs2, outputs2)

def load_data(dataset_name, **kwargs):
    """
    Loads data based on the dataset name.
    For social media data, it uses the custom loader.
    For other datasets, it might use existing loaders or raise an error.
    """
    if dataset_name == 'social_media':
        return load_social_media_data(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
