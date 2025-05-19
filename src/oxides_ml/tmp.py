def create_loaders(dataset: InMemoryDataset,
                   split: int=5,
                   batch_size: int=32,
                   test: bool=True, 
                   balance_func: callable=None) -> tuple[DataLoader]:
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
    train_loader, val_loader, test_loader = [], [], []
    n_items = len(dataset)
    sep = n_items // split
    # dataset = dataset.shuffle()
    if test:
        test_loader += (dataset[:sep])
        val_loader += (dataset[sep:sep*2])
        train_loader += (dataset[sep*2:])
    else:
        val_loader += (dataset[:sep])
        train_loader += (dataset[sep:])
    if balance_func != None:
        train_loader = balance_func(train_loader)
    # Balance gas data in training set
    gas_graphs = [graph for graph in train_loader if graph.metal == 'N/A' and graph.facet == 'N/A']
    train_loader += gas_graphs * 9
    train_n = len(train_loader)
    val_n = len(val_loader)
    test_n = len(test_loader)
    total_n = train_n + val_n + test_n
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
    if test:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
        return (train_loader, val_loader, test_loader)
    else:
        print("Data split (train/val): {}/{} %".format(int(100*(split-1)/split), int(100/split)))
        print("Training data = {} Validation data = {} (Total = {})".format(train_n, val_n, total_n))
        return (train_loader, val_loader, None)
    


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
            if graph.material in ("IrO2", "RuO2"):
                tmp_list.append(graph)
            elif graph.material in ("Ir", "Ru"):
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


def create_loaders(dataset: InMemoryDataset,
                   split: int = 5,
                   batch_size: int = 32,
                   test: bool = True,
                   balance_func: callable = None,
                   key_elements: list[str] = None,
                   key_split_ratio: float = 0.5) -> tuple[DataLoader]:
    """
    Create dataloaders for training, validation and test.
    Args:
        dataset (InMemoryDataset): Dataset object.
        split (int): Number of splits to generate train/val/test sets. Default is 5.
        batch_size (int): Batch size for each dataloader. Default is 32.
        test (bool): Whether to generate a test set in addition to train and validation. Default is True.
        balance_func (callable): Optional function to apply balancing to the training set. Default is None.
        key_elements (list[str]): List of elements (e.g. ['Ir', 'Ru']) that must be considered for partial split.
        key_split_ratio (float): Proportion of `key_elements` data to assign to training set. The rest is split 
                                 equally between validation and test. Default is 0.5.

    Returns:
        tuple: DataLoader objects for train, validation, and test sets. If `test` is False, the third output is None.
    """

    def exact_match(data, keys):
        return data.material in keys

    train_loader, val_loader, test_loader = [], [], []
    n_items = len(dataset)
    dataset = dataset.shuffle()

    # Separate key element data
    key_data = [data for data in dataset if key_elements and exact_match(data, key_elements)]
    other_data = [data for data in dataset if not (key_elements and exact_match(data, key_elements))]

    # Split key_data: e.g. 50% train, 25% val, 25% test
    n_key = len(key_data)
    n_key_train = int(key_split_ratio * n_key)
    n_key_val = int((n_key - n_key_train) // 2)
    n_key_test = n_key - n_key_train - n_key_val

    key_train = key_data[:n_key_train]
    key_val = key_data[n_key_train:n_key_train + n_key_val]
    key_test = key_data[n_key_train + n_key_val:]

    # Split other_data according to split ratio
    n_other = len(other_data)
    sep = n_other // split
    other_test = other_data[:sep] if test else []
    other_val = other_data[sep:2 * sep]
    other_train = other_data[2 * sep:] if test else other_data[sep:]

    # Combine
    train_loader = key_train + other_train
    val_loader = key_val + other_val
    test_loader = key_test + other_test if test else []

    if balance_func is not None:
        train_loader = balance_func(train_loader)

    train_n, val_n, test_n = len(train_loader), len(val_loader), len(test_loader)
    total_n = train_n + val_n + test_n

    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)

    if test:
        test_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=False)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print("Training data = {} Validation data = {} Test data = {} (Total = {})".format(train_n, val_n, test_n, total_n))
        return (train_loader, val_loader, test_loader)
    else:
        print("Data split (train/val): {}/{} %".format(int(100*(split-1)/split), int(100/split)))
        print("Training data = {} Validation data = {} (Total = {})".format(train_n, val_n, total_n))
        return (train_loader, val_loader, None)