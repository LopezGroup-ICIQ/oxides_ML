

# Define multiple hyperparameters and their respective values
hyperparams_to_test = {
    "train.lr0": [1e-6, 1e-5, 1e-4, 1e-3],          # Learning rate
    "train.patience": [6, 5, 4, 3],                 # Patience
    "train.factor": [0.9, 0.8, 0.7, 0.6],           # lr-scheduler scaling factor
    "train.minlr": [1e-6, 1e-7, 1e-8, 1e-9],        # Minimum learning rate 

    "architecture.dim": [128, 64, 32, 16],     # Layer width
    "architecture.num_linear": [0, 1 ,2, 3],        # Number of dense layers at the start of the model
    "architecture.num_conv": [2, 3, 4 ,5],       # Number of convolutional layers
    "architecture.pool_heads": [0, 1 ,2, 3],        # Number of multihead attention blocks in pooling layer
}


from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
import random

def split_percentage(split):
    if split < 2:
        return 100, 0, 0
    test_pct = 100 // split
    val_pct = test_pct
    train_pct = 100 - test_pct - val_pct
    return train_pct, val_pct, test_pct

def create_loaders(dataset: InMemoryDataset,
                   split: int = 5,
                   batch_size: int = 32,
                   test: bool = True,
                   balance_func: callable = None,
                   key_elements: list[str] = None,
                   key_split_ratio: float = 0.5) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Create dataloaders for training, validation and test.
    """

    dataset = dataset.shuffle()

    # Group data by material
    key_data_by_element = defaultdict(list)
    other_data = []

    for data in dataset:
        material = getattr(data, "material", None)
        if key_elements and material in key_elements:
            key_data_by_element[material].append(data)
        else:
            other_data.append(data)

    # Split key element data per material
    key_train, key_val, key_test = [], [], []

    for material, data_list in key_data_by_element.items():
        random.shuffle(data_list)
        n = len(data_list)
        n_train = int(n * key_split_ratio)
        remaining = n - n_train
        n_val = remaining // 2
        n_test = remaining - n_val if test else 0

        key_train.extend(data_list[:n_train])
        key_val.extend(data_list[n_train:n_train + n_val])
        if test:
            key_test.extend(data_list[n_train + n_val:])

    # Split the other data
    n_other = len(other_data)
    sep = n_other // split
    other_test = other_data[:sep] if test else []
    other_val = other_data[sep:2 * sep]
    other_train = other_data[2 * sep:] if test else other_data[sep:]

    # Combine key and other data
    train_data = key_train + other_train
    val_data = key_val + other_val
    test_data = key_test + other_test if test else []

    if balance_func:
        train_data = balance_func(train_data)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test else None

    # Logging
    train_n, val_n, test_n = len(train_data), len(val_data), len(test_data)
    total_n = train_n + val_n + (test_n if test else 0)

    if test:
        a, b, c = split_percentage(split)
        print(f"Data split (train/val/test): {a}/{b}/{c} %")
        print(f"Training = {train_n} | Validation = {val_n} | Test = {test_n} (Total = {total_n})")
    else:
        print(f"Data split (train/val): {int(100 * (split - 1) / split)}/{int(100 / split)} %")
        print(f"Training = {train_n} | Validation = {val_n} (Total = {total_n})")

    return train_loader, val_loader, test_loader
