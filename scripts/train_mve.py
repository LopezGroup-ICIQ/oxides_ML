"""Perform GNN model training with uncertainty quantification."""

import argparse
from os.path import isdir
import time
import sys
sys.path.insert(0, "../src")

import torch
torch.backends.cudnn.deterministic = True 
import toml
from torch_geometric.seed import seed_everything
seed_everything(42)
import numpy as np
from numpy import random

from oxides_ml.training import create_loaders, scale_target, train_loop, test_loop, nll_loss, nll_loss_warmup
from oxides_ml.classes import EarlyStopper
from oxides_ml.nets import GameNetUQ
from oxides_ml.post_training import create_model_report
from oxides_ml.dataset import OxidesGraphDataset

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform a training process with the provided hyperparameter settings.")
    PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the learning process.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Output directory for the results.")
    ARGS = PARSER.parse_args()
    
    output_name = ARGS.i.split("/")[-1].split(".")[0]
    output_directory = ARGS.o
    if isdir("{}/{}".format(output_directory, output_name)):
        output_name = input("There is already a model with the chosen name in the provided directory, provide a new one: ")
    
    # Upload training hyperparameters from .toml file
    hyperparameters = toml.load(ARGS.i)  
    vasp_directory = hyperparameters["data"]["vasp_directory"]
    graph_dataset_dir = hyperparameters["data"]["graph_dataset_path"]
    initial_state = hyperparameters['data']['initial_state']
    augment = hyperparameters['data']['augment']
    force_reload = hyperparameters['data']['force_reload']
    graph_settings = hyperparameters["graph"]
    train = hyperparameters["train"]
    architecture = hyperparameters["architecture"]        
    # Select device
    device_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("Device name: {} (GPU)".format(torch.cuda.get_device_name(0)))
        device_dict["name"] = torch.cuda.get_device_name(0)
        device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
        device_dict["CUDNN_version"] = torch.backends.cudnn.version()
        device_dict["CUDA_version"] = torch.version.cuda
    else:
        print("Device name: CPU")
        device_dict["name"] = "CPU"     

    # Load graph dataset 
    dataset = OxidesGraphDataset(vasp_directory, graph_dataset_dir, graph_settings, initial_state=initial_state, augment=augment, force_reload=force_reload)

    # # Filter your data 
    # filtered_data = [data for data in dataset if data.target < -0.3]

    # # Replace dataset with filtered version
    # dataset.data, dataset.slices = dataset.collate(filtered_data)

    ohe_elements = dataset.ohe_elements
    node_feature_list = dataset.node_feature_list
    num_node_features = len(node_feature_list)

    # Create train/validation/test dataloaders (apply oversampling here for gas)
    train_loader, val_loader, test_loader = create_loaders(dataset,
                                                           batch_size=train["batch_size"],
                                                           split=train["splits"], 
                                                           test=train["test_set"], 
                                                           balance_func=None) 
    
    # Target scaling 
    train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                    val_loader,
                                                                    test_loader, 
                                                                    mode=train["target_scaling"], 
                                                                    test=train["test_set"])    
    
    # Model initialization
    model = GameNetUQ(len(node_feature_list),                    
                    dim=architecture["dim"],
                    num_linear=architecture["num_linear"], 
                    num_conv=architecture["num_conv"],    
                    bias=architecture["bias"],
                    uq = architecture['uq']).to(device)
    initial_params = {name: p.clone() for name, p in model.named_parameters()}  

    # Load optimizer, lr-scheduler, and early stopper
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train["lr0"],
                                 eps=train["eps"], 
                                 weight_decay=train["weight_decay"],
                                 amsgrad=train["amsgrad"])
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=train["factor"],
                                                              patience=train["patience"],
                                                              min_lr=train["minlr"])
    
    if train["early_stopping"]:
        early_stopper = EarlyStopper(patience=train["es_patience"], start_epoch=train["es_start_epoch"])       
    
    # Training loop
    loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []
    train_std, val_std, test_std = [], [], [] 
    t0 = time.time()
    for epoch in range(1, train["epochs"]+1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]['lr']        
        loss_func = nll_loss if epoch > 0 else nll_loss_warmup
        loss, train_MAE, train_scale = train_loop(model, device, train_loader, optimizer, loss_func)  
        val_MAE, val_scale = test_loop(model, val_loader, device, std)  
        lr_scheduler.step(val_MAE)
        if train["test_set"]:
            test_MAE, test_scale = test_loop(model, test_loader, device, std)         
            test_list.append(test_MAE)
            test_std.append(test_scale)
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Val MAE: {:.4f} eV '             
                  'Test MAE: {:.4f} eV'.format(epoch, lr, train_MAE*std, val_MAE, test_MAE))
        else:
            print('Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Val MAE: {:.6f} eV '
                  .format(epoch, lr, train_MAE*std, val_MAE))         
        loss_list.append(loss)
        train_list.append(train_MAE * std)
        train_std.append(train_scale * std)
        val_list.append(val_MAE)
        val_std.append(val_scale)
        lr_list.append(lr)
        if train["early_stopping"]:
            if early_stopper.stop(val_MAE, epoch):
                print("Early stopping at epoch {}.".format(epoch))
                break
    parameter_changes = {}
    for name, p in model.named_parameters():
        change = torch.norm(p - initial_params[name])  # L2 norm
        parameter_changes[name] = change
    sorted_changes = sorted(parameter_changes.items(), key=lambda x: x[1], reverse=True)
    print("-----------------------------------------------------------------------------------------")
    training_time = (time.time() - t0) / 60  
    print("Training time: {:.2f} min".format(training_time))
    device_dict["training_time"] = training_time
    create_model_report(output_name,
                        output_directory,
                        hyperparameters,  
                        model, 
                        (train_loader, val_loader, test_loader),
                        (mean, std),
                        (train_list, val_list, test_list, lr_list),
                        ohe_elements, 
                        device_dict, 
                        parameter_changes, 
                        (train_std, val_std, test_std))
