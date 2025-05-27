"""
This module contains functions used for training and testing the GNN models.
"""

import math
from copy import copy, deepcopy

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
import random
import hashlib

## Fixed loaders
import hashlib
from torch_geometric.data import DataLoader, InMemoryDataset

def assign_split(graph, split: int) -> str:
    """
    Assign a graph to train/val/test deterministically using a hash of its ID.
    Accepts graph_id as a string or a 0-D tensor (e.g., torch.Tensor([123])).
    """
    assert split >= 3, "Split must be at least 3 to allow train/val/test"

    # Handle Tensor IDs (e.g., tensor(123))
    graph_id = str(graph.graph_id.item())

    hash_val = hashlib.md5(graph_id.encode()).hexdigest()
    hash_int = int(hash_val, 16)
    bucket = hash_int % split

    if bucket == 0:
        return 'test'
    elif bucket == 1:
        return 'val'
    else:
        return 'train'

def create_loaders_db1_fixed(dataset: InMemoryDataset,
                       split: int = 5,
                       batch_size: int = 32,
                       key_elements: list[str] = None
                       ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and test using stable hash-based ID splits.

    Args:
        dataset (InMemoryDataset): PyG dataset.
        split (int): Number of splits (1/split for test and val).
        batch_size (int): Batch size for loaders.
        key_elements (list[str], optional): List of key elements that must be in training set (not enforced).

    Returns:
        tuple: DataLoader objects for train, validation, and test sets.
    """
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if getattr(graph, "type", None) != "slab":
            if getattr(graph, "material", None) in ("IrO2", "RuO2", "TiO2"):
                split_label = assign_split(graph, split)
                if split_label == 'train':
                    train_list.append(graph)
                elif split_label == 'val':
                    val_list.append(graph)
                elif split_label == 'test':
                    test_list.append(graph)

    random.shuffle(train_list)
    random.shuffle(val_list)    
    random.shuffle(test_list)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    total_n = len(train_list) + len(val_list) + len(test_list)
    print(f"Training data = {len(train_list)} | Validation data = {len(val_list)} | Test data = {len(test_list)} | Total = {total_n}")

    return train_loader, val_loader, test_loader

def create_loaders_db2_fixed(dataset: InMemoryDataset,
                       split: int = 5,
                       batch_size: int = 32,
                       key_elements: list[str] = None
                       ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation and test using hash-based split logic.
    
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        key_elements (list[str]): List of elements (e.g. ['Ir', 'Ru']) that must be in the training set.
    
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if getattr(graph, "type", None) != "slab":
            mat = getattr(graph, "material", None)

            if key_elements is not None:
                if mat in key_elements:
                    train_list.append(graph)  # Force into training set
                elif mat in ("IrO2", "RuO2", "TiO2"):
                    split_label = assign_split(graph, split)
                    if split_label == 'train':
                        train_list.append(graph)
                    elif split_label == 'val':
                        val_list.append(graph)
                    elif split_label == 'test':
                        test_list.append(graph)
            else:
                if mat in ("IrO2", "RuO2", "TiO2", "Ir", "Ru", "Ti"):
                    split_label = assign_split(graph, split)
                    if split_label == 'train':
                        train_list.append(graph)
                    elif split_label == 'val':
                        val_list.append(graph)
                    elif split_label == 'test':
                        test_list.append(graph)

    random.shuffle(train_list)  # Shuffle for training batches
    random.shuffle(val_list)  # Shuffle for validation batches
    random.shuffle(test_list)  # Shuffle for test batches

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    total_n = len(train_list) + len(val_list) + len(test_list)
    print(f"Training data = {len(train_list)} | Validation data = {len(val_list)} | Test data = {len(test_list)} | Total = {total_n}")

    return train_loader, val_loader, test_loader

def create_loaders_db3_fixed(dataset: InMemoryDataset,
                       split: int = 5,
                       batch_size: int = 32,
                       key_elements: list[str] = None
                       ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation and test using hash-based ID splitting.

    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        key_elements (list[str]): Ignored in this version, but included for signature consistency.

    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if getattr(graph, "type", None) != "slab":
            split_label = assign_split(graph, split)
            if split_label == 'train':
                train_list.append(graph)
            elif split_label == 'val':
                val_list.append(graph)
            elif split_label == 'test':
                test_list.append(graph)

    random.shuffle(train_list)  # Shuffle training set for batching
    random.shuffle(val_list)  # Shuffle validation set for consistency
    random.shuffle(test_list)  # Shuffle test set for consistency

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    total_n = len(train_list) + len(val_list) + len(test_list)
    print(f"Training data = {len(train_list)} | Validation data = {len(val_list)} | Test data = {len(test_list)} | Total = {total_n}")

    return train_loader, val_loader, test_loader

## Random loaders
def split_percentage(splits: int, test: bool=True) -> tuple[int]:
    """Return split percentage of the train, validation and test sets.
    One split represent the test set, one the validation set and the rest the train set.
    Args:
        split (int): number of initial splits of the entire initial dataset

    Returns:
        a, b, c: train, validation, test percentage of the sets.
    Examples:
        >>> split_percentage(5) # 5 splits
        (60, 20, 20)  # 60% of data for training, 20% for validation and 20% for testing
        >>> split_percentage(5, test=False) # 5 splits
        (80, 20, 0)  # 80% of data for training, 20% for validation and 0% for testing
    """
    if test:
        a = int(100 - 200 / splits)
        b = math.ceil(100 / splits)
        return a, b, b
    else:
        return int((1 - 1/splits) * 100), math.ceil(100 / splits)


def create_loaders_db1(dataset: InMemoryDataset,
                   split: int = 5,
                   batch_size: int = 32,
                   key_elements: list[str] = None,
                   ) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        key_elements (list[str]): List of elements (e.g. ['Ir', 'Ru']) that must be in the training set.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                tmp_list.append(graph)

    # Split test and validation according to split
    random.shuffle(tmp_list)
    n_items = len(tmp_list)
    sep = n_items // split
    test_list = tmp_list[:sep]
    val_list = tmp_list[sep:sep*2]
    train_list = tmp_list[sep*2:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)
    
def create_loaders_db2(dataset: InMemoryDataset,
                   split: int = 5,
                   batch_size: int = 32,
                   key_elements: list[str] = None,
                   ) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        key_elements (list[str]): List of elements (e.g. ['Ir', 'Ru']) that must be in the training set.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    if key_elements is not None:
        for graph in dataset:
            if graph.type not in ("slab"):
                if graph.material in (key_elements):
                    train_list.append(graph)
                elif graph.material in ("IrO2", "RuO2", "TiO2"):
                    tmp_list.append(graph)
    else:
        for graph in dataset:
            if graph.type not in ("slab"):
                if graph.material in ("IrO2", "RuO2", "TiO2", "Ir", "Ru", "Ti"):
                    tmp_list.append(graph)

    # Split test and validation according to split
    random.shuffle(tmp_list)
    n_items = len(tmp_list + train_list)
    sep = n_items // split
    test_list = tmp_list[:sep]
    val_list = tmp_list[sep:sep*2]

    # Add the remaining tmp_list data to train_list
    train_list += tmp_list[sep*2:]
    random.shuffle(train_list)

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_db3(dataset: InMemoryDataset,
                   split: int = 5,
                   batch_size: int = 32,
                   key_elements: list[str] = None,
                   ) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        key_elements (list[str]): List of elements (e.g. ['Ir', 'Ru']) that must be in the training set.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            tmp_list.append(graph)

    # Split test and validation according to split
    random.shuffle(tmp_list)
    n_items = len(dataset)
    sep = n_items // split
    test_list = tmp_list[:sep]
    val_list = tmp_list[sep:sep*2]
    train_list = tmp_list[sep*2:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)
    
def create_loaders_exp1(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                test_list.append(graph)
            elif graph.material in ("Ir", "Ru", "Ti"):
                train_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp2(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                test_list.append(graph)
            else:
                train_list.append(graph)
                
    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)
    
def create_loaders_exp3(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                train_list.append(graph)
            elif graph.material in ("Ir", "Ru", "Ti"):
                test_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)


def create_loaders_exp4(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                tmp_list.append(graph)
            elif graph.material in ("Ir", "Ru", "Ti"):
                train_list.append(graph)

    # take randomly 50% of tmp_list data and put it in test_list
    random.shuffle(tmp_list)
    n_tmp = len(tmp_list)
    n_test = int(n_tmp * 0.5)
    test_list = tmp_list[:n_test]
    tmp_list = tmp_list[n_test:]

    train_list += tmp_list

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp5(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material == ("TiO2"):
                test_list.append(graph)
            elif graph.material in ("IrO2", "RuO2", "Ir", "Ru", "Ti"):
                train_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp6(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material == ("TiO2"):
                test_list.append(graph)
            elif graph.material in ("IrO2", "RuO2"):
                train_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp6b(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material == ("RuO2"):
                test_list.append(graph)
            elif graph.material in ("IrO2", "TiO2", "Ir", "Ru", "Ti"):
                train_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp6c(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material == ("IrO2"):
                test_list.append(graph)
            elif graph.material in ("RuO2", "TiO2", "Ir", "Ru", "Ti"):
                train_list.append(graph)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp7(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("IrO2", "RuO2", "TiO2"):
                tmp_list.append(graph)


    # take randomly 50% of tmp_list data and put it in test_list
    random.shuffle(tmp_list)
    n_tmp = len(tmp_list)
    n_test = int(n_tmp * 0.5)
    test_list = tmp_list[:n_test]
    tmp_list = tmp_list[n_test:]

    train_list += tmp_list    

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp7b(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.dissociation == False:
                if graph.material in ("IrO2", "RuO2", "TiO2"):
                    tmp_list.append(graph)


    # take randomly 50% of tmp_list data and put it in test_list
    random.shuffle(tmp_list)
    n_tmp = len(tmp_list)
    n_test = int(n_tmp * 0.5)
    test_list = tmp_list[:n_test]
    tmp_list = tmp_list[n_test:]

    train_list += tmp_list    

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp8(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material not in ("IrO2", "RuO2", "TiO2"):
                tmp_list.append(graph)

    # take randomly 50% of tmp_list data and put it in test_list
    random.shuffle(tmp_list)
    n_tmp = len(tmp_list)
    n_test = int(n_tmp * 0.5)
    test_list = tmp_list[:n_test]
    tmp_list = tmp_list[n_test:]

    train_list += tmp_list  

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp9(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list, tmp_list = [], [], [], []

    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("Ir", "Ru", "Ti"):
                tmp_list.append(graph)

    # take randomly 50% of tmp_list data and put it in test_list
    random.shuffle(tmp_list)
    n_tmp = len(tmp_list)
    n_test = int(n_tmp * 0.5)
    test_list = tmp_list[:n_test]
    tmp_list = tmp_list[n_test:]

    train_list += tmp_list    

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def create_loaders_exp10(dataset: InMemoryDataset,
                   batch_size: int=32, **kwargs) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset : Dataset object.
        split (int): number of splits to generate train/val/test sets. Default to 5.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate test set besides train and val sets. Default to True.   
        balance_func (callable): function to balance the training set. Default to None.
    Returns:
        (tuple): DataLoader objects for train, validation and test sets.
    """
    dataset = dataset.shuffle()
    train_list, val_list, test_list = [], [], []

    grouped_graphs = defaultdict(list)
    for graph in dataset:
        if graph.type not in ("slab"):
            if graph.material in ("TiO2", "IrO2", "RuO2"):
                key = (graph.material, graph.adsorbate_name)
                grouped_graphs[key].append(graph)

    # Select most stable (most negative ads_energy) per group
    for group in grouped_graphs.values():
        # Find graph with most negative adsorption energy
        most_stable = min(group, key=lambda g: g.ads_energy)
        train_list.append(most_stable)
        # Add the rest to test list
        for g in group:
            if g is not most_stable:
                test_list.append(g)

    # take randomly 20% of train_list data and put it in val_list
    random.shuffle(train_list)
    n_train = len(train_list)
    n_val = int(n_train * 0.2)
    val_list = train_list[:n_val]
    train_list = train_list[n_val:]

    train_n = len(train_list)
    val_n = len(val_list)
    test_n = len(test_list)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)
    print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
    return (train_loader, val_loader, test_loader)

def scale_target(train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader=None,
                 mode: str='std',
                 verbose: bool=True,
                 test: bool=True):
    """
    Apply target scaling to the whole dataset using training and validation sets.
    Args:
        train_loader (torch_geometric.loader.DataLoader): training dataloader 
        val_loader (torch_geometric.loader.DataLoader): validation dataloader
        test_loader (torch_geometric.loader.DataLoader): test dataloader
    Returns:
        train, val, test: dataloaders with scaled target values
        mean_tv, std_tv: mean and std (standardization)
        min_tv, max_tv: min and max (normalization)
    """
    # 1) Get target scaling coefficients from train and validation sets
    y_list = []
    for graph in train_loader.dataset:
        y_list.append(graph.target.item())
    for graph in val_loader.dataset:
        y_list.append(graph.target.item())
    y_tensor = torch.tensor(y_list)
    # Standardization
    mean_tv = y_tensor.mean(dim=0, keepdim=True)  
    std_tv = y_tensor.std(dim=0, keepdim=True)
    # Normalization
    max_tv = y_tensor.max()
    min_tv = y_tensor.min()
    delta_norm = max_tv - min_tv
    # 2) Apply Scaling
    for graph in train_loader.dataset:
        if mode == "std":
            graph.y = (graph.target - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.target - min_tv) / (max_tv - min_tv)
        elif mode == "none":
            graph.y = graph.target
        else:
            pass
    for graph in val_loader.dataset:
        if mode == "std":
            graph.y = (graph.target - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.target - min_tv) / delta_norm
        elif mode == "none":
            graph.y = graph.target
        else:
            pass
    if test:
        for graph in test_loader.dataset:
            if mode == "std":
                graph.y = (graph.target - mean_tv) / std_tv
            elif mode == "norm":
                graph.y = (graph.target - min_tv) / delta_norm
            elif mode == "none":
                graph.y = graph.target
            else:
                pass
    if mode == "std":
        if verbose:
            print("Target Scaling (Standardization) applied successfully")
            print("(Train+Val) mean: {:.2f} eV".format(mean_tv.item()))
            print("(Train+Val) standard deviation: {:.2f} eV".format(std_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, mean_tv.item(), std_tv.item()
        else:
            return train_loader, val_loader, None, mean_tv.item(), std_tv.item()
    elif mode == "norm": 
        if verbose:
            print("Target Scaling (Normalization) applied successfully")
            print("(Train+Val) min: {:.2f} eV".format(min_tv.item()))
            print("(Train+Val) max: {:.2f} eV".format(max_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, min_tv.item(), max_tv.item()
        else:
            return train_loader, val_loader, None, min_tv.item(), max_tv.item()
    else:
        print("Target Scaling not applied")
        if test:
            return train_loader, val_loader, test_loader, 0, 1
        else:
            return train_loader, val_loader, None, 0, 1


def train_loop(model: torch.nn.Module,
               device:str,
               train_loader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn) -> tuple[float]:
    """
    Run training iteration (epoch) 
    For each batch in the epoch, the following actions are performed:
    1) Move the batch to the training device
    2) Forward pass through the GNN model and compute loss
    3) Compute gradient of loss function wrt model parameters
    4) Update model parameters
    Args:
        model(): GNN model object.
        device(str): device on which training is performed.
        train_loader(): Training dataloader.
        optimizer(): optimizer used during training.
        loss_fn(): Loss function used for the training.
    Returns:
        loss_all, mae_all (tuple[float]): Loss function and MAE of the whole epoch.   
    """
    model.train()  
    loss_all, mae_all = 0, 0
    mean_std = 0
    for batch in train_loader:  # batch-wise
        batch = batch.to(device)
        optimizer.zero_grad()                     # Set gradients of all tensors to zero
        loss = loss_fn(model, batch)
        mae = F.l1_loss(model(batch).mean.squeeze(), batch.y)    # For comparison with val/test data
        mean_std += model(batch).scale.sum().item()
        loss.backward()                           # Get gradient of loss function wrt parameters
        loss_all += loss.item() * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs
        optimizer.step()                          # Update model parameters
    loss_all /= len(train_loader.dataset)
    mae_all /= len(train_loader.dataset)
    mean_std /= len(train_loader.dataset)
    return loss_all, mae_all, mean_std


def test_loop(model: torch.nn.Module,
              loader: DataLoader,
              device: str,
              std: float) -> tuple[float]:
    """
    Run test/validation iteration (epoch).
    For each batch of validation/test data, the following steps are performed:
    1) Move the batch to the training device
    2) Compute the Mean Absolute Error (MAE) and mean standard deviation of the batch
    Args:
        model (): GNN model object.
        loader (Dataloader object): Dataset for validation/testing.
        device (str): device on which training is performed.
        std (float): standard deviation of the training+validation datasets [eV]
    Returns:
        error, std_mean (tuple[float]): MAE and mean standard deviation of the whole val/test set.
    """
    model.eval()       
    error, std_mean = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            error += (model(batch).mean * std - batch.y * std).abs().sum().item()   # eV
            std_mean += model(batch).scale.sum().item() * std  # eV
    return error / len(loader.dataset), std_mean / len(loader.dataset) 


def get_mean_std_from_model(path:str) -> tuple[float]:
    """Get mean and standard deviation used for scaling the target values 
       from the selected trained model.

    Args:
        model_name (str): GNN model path.
    
    Returns:
        mean, std (tuple[float]): mean and standard deviation for scaling the targets.
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "(train+val) mean" in line:
            mean = float(line.split()[-2])
        if "(train+val) standard deviation" in line:
            std = float(line.split()[-2])
    return mean, std


def get_graph_conversion_params(path: str) -> tuple:
    """Get the hyperparameters for geometry->graph conversion algorithm.
    Args:
        path (str): path to directory containing the GNN model.
    Returns:
        tuple: voronoi tolerance (float), scaling factor (float), metal nearest neighbours inclusion (bool)
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "Voronoi" in line:
            voronoi_tol = float(line.split()[-2])
        if "scaling factor" in line:
            scaling_factor = float(line.split()[-1])
        if "Second order" in line:
            if line.split()[-1] == "True":
                second_order_nn = True
            else:
                second_order_nn = False
    return voronoi_tol, scaling_factor, second_order_nn 


def split_list(a: list, n: int):
    """
    Split a list into n chunks (for nested cross-validation)
    Args:
        a(list): list to split
        n(int): number of chunks
    Returns:
        (list): list of chunks
    """
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def create_loaders_nested_cv(dataset: InMemoryDataset, 
                             split: int, 
                             batch_size: int):
    """
    Create dataloaders for training, validation and test sets for nested cross-validation.
    Args:
        datasets(tuple): tuple containing the HetGraphDataset objects.
        split(int): number of splits to generate train/val/test sets
        batch(int): batch size    
    Returns:
        (tuple): tuple with dataloaders for training, validation and testing.
    """
    # Create list of lists, where each list contains the datasets for a split.
    chunk = [[] for _ in range(split)]
    
    dataset.shuffle()
    iterator = split_list(dataset, split)
    for index, item in enumerate(iterator):
        chunk[index] += item
    chunk = sorted(chunk, key=len)
    # Create dataloaders for each split.    
    for index in range(len(chunk)):
        proxy = copy(chunk)
        test_loader = DataLoader(proxy.pop(index), batch_size=batch_size, shuffle=False)
        for index2 in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy(proxy)
            val_loader = DataLoader(proxy2.pop(index2), batch_size=batch_size, shuffle=False)
            flatten_training = [item for sublist in proxy2 for item in sublist]  # flatten list of lists
            train_loader = DataLoader(flatten_training, batch_size=batch_size, shuffle=True)
            yield deepcopy((train_loader, val_loader, test_loader))

def nll_loss(model, data):
    """
    Negative log likelihood loss function."""
    normal_dist = model(data)
    neg_log_likelihood = -normal_dist.log_prob(data.y)
    return torch.mean(neg_log_likelihood)
    
def nll_loss_warmup(model, data, device):
    normal_dist = model(data)
    normal_dist.scale = torch.tensor(1e-3).to(device)
    neg_log_likelihood = -normal_dist.log_prob(data.y)
    return torch.mean(neg_log_likelihood)


def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
