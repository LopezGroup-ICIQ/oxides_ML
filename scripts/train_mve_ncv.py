"""Perform nested cross validation for GNN using the FG-dataset.
To run it, you can use the following command:
python nested_cross_validation_GNN.py -i input_hyperparams.toml -o ../results_dir
The input toml has the same structure as the one used for the training process."""

import argparse
from os.path import exists, isdir
from os import mkdir, listdir
import time
import sys
sys.path.insert(0, "../src")

import torch 
import toml
import numpy as np
import pandas as pd

from oxides_ml.functions import  create_loaders_nested_cv
from oxides_ml.constants import loss_dict
from oxides_ml.classes import EarlyStopper
from oxides_ml.nets import GameNetUQ
from oxides_ml.post_training import create_model_report
from oxides_ml.dataset import OxidesGraphDataset
from oxides_ml.training_TiO2 import nll_loss, nll_loss_warmup, scale_target, train_loop, test_loop


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Perform nested cross validation for GNN using the FG-dataset.")
    PARSER.add_argument("-i", "--input", type=str, dest="i", 
                        help="Input toml file with hyperparameters for the nested cross validation.")
    PARSER.add_argument("-o", "--output", type=str, dest="o", 
                        help="Directory where the results will be saved.")
    ARGS = PARSER.parse_args()
    
    output_name = ARGS.i.split("/")[-1].split(".")[0]
    output_directory = ARGS.o
    if isdir("{}/{}".format(output_directory, output_name)):
        output_name = input("There is already a model with the chosen name in the provided directory, provide a new one: ")
    mkdir("{}/{}".format(output_directory, output_name))

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
        
    print("Nested cross validation for GNN using the FG-dataset")
    print("Number of splits: {}".format(train["splits"]))
    print("Total number of runs: {}".format(train["splits"]*(train["splits"]-1)))
    print("--------------------------------------------")
    # Select device (GPU/CPU)
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

    ohe_elements = dataset.ohe_elements
    node_feature_list = dataset.node_feature_list
    num_node_features = len(node_feature_list)

    # Dataset contains all the data, filter to create the different databases
    tmp_list = []

    # # Database 1: IrO2, RuO2, and TiO2
    # for graph in dataset:
    #     if graph.type not in ("slab"):
    #         if graph.material in ("IrO2", "RuO2", "TiO2"):
    #             tmp_list.append(graph)

    # # Database 2: All points
    # for graph in dataset:
    #     if graph.type not in ("slab"):
    #         if graph.material in ("IrO2", "RuO2", "Ru", "Ir"):
    #             tmp_list.append(graph)

    # Database 3: All points
    for graph in dataset:
        if graph.type not in ("slab", "gas"):
            if graph.material not in ("TiO2", "Ti"):
                tmp_list.append(graph)

    dataset = tmp_list

    # Instantiate iterator for nested cross validation: Each iteration yields a different train/val/test set combination
    ncv_iterator = create_loaders_nested_cv(dataset, split=train["splits"], batch_size=train["batch_size"])        
    MAE_outer = []
    counter = 0
    TOT_RUNS = train["splits"]*(train["splits"]-1)
    for outer in range(train["splits"]):
        MAE_inner = []
        for inner in range(train["splits"]-1):
            counter += 1
            train_loader, val_loader, test_loader = next(ncv_iterator)
            train_loader, val_loader, test_loader, mean, std = scale_target(train_loader,
                                                                            val_loader, 
                                                                            test_loader,
                                                                            mode=train["target_scaling"],
                                                                            test=True)  # True is necessary condition for nested CV

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

            loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []   
            train_std, val_std, test_std = [], [], []       
            t0 = time.time()
            # Run the learning    
            for epoch in range(1, train["epochs"]+1):
                torch.cuda.empty_cache()
                lr = lr_scheduler.optimizer.param_groups[0]['lr']     
                loss_func = nll_loss if epoch > 0 else nll_loss_warmup   
                loss, train_MAE, train_scale = train_loop(model, device, train_loader, optimizer, loss_func)  
                val_MAE, val_scale = test_loop(model, val_loader, device, std)  
                lr_scheduler.step(val_MAE)
                test_MAE, test_scale = test_loop(model, test_loader, device, std)         
                print('{}/{}-Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV '             
                      'Test MAE: {:.4f} eV'.format(counter, TOT_RUNS, epoch, lr, train_MAE*std, val_MAE, test_MAE))
                test_list.append(test_MAE)
                test_std.append(test_scale)       
                loss_list.append(loss)
                train_list.append(train_MAE * std)
                train_std.append(train_scale * std)
                val_list.append(val_MAE)
                val_std.append(val_scale)
                lr_list.append(lr)
                if epoch == train["epochs"]:
                    MAE_inner.append(test_MAE)
            print("-----------------------------------------------------------------------------------------")
            training_time = (time.time() - t0)/60  
            print("Training time: {:.2f} min".format(training_time))
            device_dict["training_time"] = training_time
            create_model_report("{}_{}".format(outer+1, inner+1),
                                output_directory+"/"+output_name,
                                hyperparameters,  
                                model, 
                                (train_loader, val_loader, test_loader),
                                (mean, std),
                                (train_list, val_list, test_list, lr_list),
                                ohe_elements, 
                                device_dict,
                                std_lists=(train_std, val_std, test_std),)
            del model, optimizer, lr_scheduler, train_loader, val_loader, test_loader
            if device == "cuda":
                torch.cuda.empty_cache()
        MAE_outer.append(np.mean(MAE_inner))
    MAE = np.mean(MAE_outer)
    print("Nested CV MAE: {:.4f} eV".format(MAE))
    # Generate report of the whole experiment
    ncv_results = listdir(output_directory + "/" + output_name)
    df = pd.DataFrame()
    for run in ncv_results:
        results = pd.read_csv(output_directory + "/" + output_name + "/" + run + "/test_set.csv", sep="\t")
        # add column with run number
        results["run"] = run
        df = pd.concat([df, results], axis=0)
    df = df.reset_index(drop=True)
    df.to_csv(output_directory + "/" + output_name + "/summary.csv", index=False)
