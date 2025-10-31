# Models Directory

This directory contains trained machine learning models, graph datasets, data loaders, and experimental outputs for the GAME-Net-Ox project.

## Directory Structure Overview

```
models/
├── DATALOADERS/                    # Pre-computed PyTorch data loaders
├── Experiments/                    # Experimental configurations and outputs
├── NCV/                            # Nested Cross-Validation (NCV) results
├── hyperparameter_optimization/    # Hyperparameter search experiments
├── test_graph_datasets/            # Processed graph datasets for training/testing
└── test_training/                  # Training runs and model outputs
```

---

## Directory Descriptions

### `DATALOADERS/`

**Purpose:** Pre-computed PyTorch DataLoader objects for efficient batch training.

**Contents:**
- `AUGMENT/`: Data loaders with data augmentation enabled
- `RELAXED/`: Data loaders using relaxed (DFT-optimized) structures
- `exclude_low_ads/`: Filtered data loaders excluding low adsorption energy samples

**Use Case:** 
- Speed up training by avoiding dataset recreation
- Ensure reproducible data splits across runs
- Share standardized data pipelines across team members

**Files Inside Each Subdirectory:**
- `*.pth`: Serialized PyTorch DataLoader objects (train, validation, test)
- `dataset_config.txt`: Configuration parameters used to create loaders

**Example Usage:**
```python
import torch
train_loader = torch.load('DATALOADERS/AUGMENT/train_loader.pth')
val_loader = torch.load('DATALOADERS/AUGMENT/val_loader.pth')
```

---

### `Experiments/`

**Purpose:** Experimental runs with different configurations and settings.

**Contents:**
- `AUGMENT/`: Experiments using augmented training data
- `RELAXED/`: Experiments using relaxed structures
- `exclude_low_ads/`: Experiments with filtered low-energy adsorbates

**Use Case:**
- Test different model architectures
- Validate data preprocessing choices
- Compare performance across variants

**Typical Output Structure Per Experiment:**
```
experiment_name/
├── model.pth                    # Final trained model weights
├── training.csv                 # Training epoch-by-epoch metrics
├── test_set.csv                 # Test set predictions and errors
├── validation_set.csv           # Validation set results
├── performance.txt              # Summary metrics (MAE, RMSE, R²)
└── params_changes.txt           # Configuration parameters used
```

**Files Generated:**
- `model.pth`: Trained model weights (PyTorch state dict)
- `training.csv`: Training logs with epoch, loss, MAE by split
- `test_set.csv`: Predictions, true values, and errors on test set
- `validation_set.csv`: Validation set performance
- `performance.txt`: Final performance metrics summary
- `input.txt`: TOML configuration used for this run

---

### `NCV/` (Nested Cross-Validation)

**Purpose:** Results from nested cross-validation (NCV) model evaluation.

**Use Case:**
- Rigorous model validation avoiding data leakage
- Hyperparameter optimization within nested folds
- Robust performance estimation without contamination

**Structure:**
```
NCV/
├── Db1/                  # Database 1 NCV results
│   ├── 1_1/              # Fold 1, Iteration 1
│   │   ├── model.pth
│   │   ├── training.csv
│   │   ├── test_set.csv
│   │   └── ...
│   ├── 1_2/
│   └── summary.csv       # Aggregated results across all folds
├── Db1_TiO2/             # Database 1, TiO2 specific
├── Db2/                  # Database 2 NCV results
├── Db2_TiO2/
├── Db3/                  # Database 3 NCV results
├── Db3_TiO2/
└── NCV tests/            # Experimental NCV configurations
```

**Key Files:**
- `summary.csv`: Aggregated NCV results (mean, std of metrics across folds)
- `*/training.csv`: Per-fold training logs
- `*/test_set.csv`: Per-fold test predictions
- `*/model.pth`: Per-fold trained model

**Naming Convention:**
- `Db1`, `Db2`, `Db3`: Database 1, 2, 3
- `_TiO2`: Material-specific variant (TiO2 oxides)
- `3_4`, `5_1`: Fold number and iteration

**Interpretation:**
- Higher-level folders: Different database sources
- Subfolder `X_Y`: Cross-validation fold X, iteration Y
- `summary.csv`: Final reported metrics (average across all folds)

---

### `hyperparameter_optimization/`

**Purpose:** Results from hyperparameter tuning experiments.

**Contents:**
- `BS_database_1/`: Batch size optimization for Database 1
- `CN_database_1/`: Coordination number feature impact for Database 1
- `initial_test/`: Preliminary hyperparameter exploration
- `test/`: Test runs for optimization framework

**Typical Structure:**
```
hyperparameter_optimization/
└── BS_database_1/
    ├── run_1/                # Batch size = 32
    │   ├── model.pth
    │   ├── training.csv
    │   ├── test_set.csv
    │   └── params_changes.txt
    ├── run_2/                # Batch size = 64
    └── summary.csv           # Comparison of all runs
```

**Use Case:**
- Identify optimal learning rate, batch size, hidden dimensions
- Explore feature importance (e.g., with/without CN)
- Document performance trade-offs

**Example Files:**
- `params_changes.txt`: Changed parameters for this run
- `summary.csv`: Comparison table across all hyperparameter values

---

### `test_graph_datasets/`

**Purpose:** Processed PyTorch Geometric graph datasets ready for training.

**Contents:**

Named datasets following the convention: `{SOURCE}_{TARGET}_{TOLERANCE}_{SCALING}_{SURFACE_ORDER}_{FEATURES}_{STATE}_{AUGMENT}`

**Example Dataset Names:**
- `database_1_adsorption_energy_025_125_3_False_False_False_True_False_False_relaxed_True`
- `database_3_adsorption_energy_03_125_2_False_False_False_True_False_False_relaxed_False`

**Structure:**
```
test_graph_datasets/
├── TiO2_adsorption_energy_*          # TiO2-specific datasets
├── database_1_adsorption_energy_*    # Database 1 graphs
├── database_2_adsorption_energy_*    # Database 2 graphs
├── database_3_adsorption_energy_*    # Database 3 graphs
├── metal_adsorbates_*                # Metal adsorbate graphs
├── oxide_adsorbates_*                # Oxide adsorbate graphs
└── processed/                        # Final datasets used for training
```

**Dataset Naming Components:**
| Position | Variable | Example | Description |
|----------|----------|---------|-------------|
| 1 | Source | `database_3` | VASP data source |
| 2 | Target | `adsorption_energy` | Prediction target |
| 3 | Tolerance | `03` | Nearest-neighbor cutoff (0.3 Å) |
| 4 | Scaling | `125` | Unit cell scaling (1.25x) |
| 5 | Surface Order | `2` | Surface layer order |
| 6-10 | Features | `False_False_False_True_False_False` | adsorbate, radical, valence, cn, magnetization, ads_height |
| 11 | State | `relaxed` | initial or relaxed structures |
| 12 | Augment | `False` | Data augmentation flag |

**File Contents:**
- Each dataset is a serialized `torch_geometric.data.InMemoryDataset`
- Contains graph representations of adsorbate/surface systems
- Ready to load directly for training: `dataset = torch.load('dataset_name')`

**For Full Dataset Documentation:** See `data/graph_datasets/README.md`

---

### `test_training/`

**Purpose:** Trained model outputs organized by database and configuration.

**Structure:**
```
test_training/
├── database_1/
│   ├── AUGMENT/              # Data augmentation enabled runs
│   │   ├── ads_height_test_1/
│   │   ├── ads_height_test_2/
│   │   ├── cn_std_3_dim_256_2/
│   │   └── ...
│   └── (other configurations)
├── database_2/
│   ├── AUGMENT/
│   │   ├── surface_order_2_cn/
│   │   └── ...
│   └── (other configurations)
├── database_3/
│   ├── AUGMENT/
│   │   ├── test_cn/
│   │   └── ...
│   ├── segmented_test/
│   └── (other configurations)
├── database_4/
├── test/                     # Quick tests and debugging runs
└── try/                      # Experimental configurations
```

**Output Structure Per Training Run:**
```
training_run/
├── model.pth                        # Final model weights
├── GNN.pth                          # Graph neural network weights
├── one_hot_encoder_elements.pth     # Element one-hot encoder (pickle)
├── training.csv                     # Epoch-by-epoch metrics
├── train_set.csv                    # Training set predictions
├── test_set.csv                     # Test set predictions
├── validation_set.csv               # Validation set predictions
├── uq.csv                           # Uncertainty quantification results (optional)
├── train_loader.pth                 # Serialized PyTorch DataLoader (train)
├── val_loader.pth                   # Serialized PyTorch DataLoader (validation)
├── test_loader.pth                  # Serialized PyTorch DataLoader (test)
├── input.txt                        # Configuration file (original TOML)
├── params_changes.txt               # Modified parameters
├── performance.txt                  # Summary metrics
├── device.txt                       # Device used (CPU/GPU)
└── Outliers/                        # Outlier analysis (optional)
    ├── largest_errors.csv
    ├── visualization.png
    └── ...
```

**Key CSV Files Format:**

**`training.csv`:**
- Columns: `Epoch, Train_Loss, Train_MAE_eV, Val_MAE_eV, Test_MAE_eV, R2_Score, ...`
- One row per epoch during training
- Used for learning curve analysis

**`test_set.csv`:**
- Columns: `Formula, Material, Adsorbate, True_energy_eV, Predicted_energy_eV, Abs_error_eV, ...`
- One row per test sample
- Complete prediction results

**`uq.csv`:**
- Columns: `True_energy_eV, Predicted_energy_eV, Predicted_std, Abs_error_eV, ...`
- Uncertainty quantification results (if enabled)

**Model Weights:**
- `model.pth`: Main model state dict (PyTorch)
- `GNN.pth`: Graph neural network weights (saved separately)
- `one_hot_encoder_elements.pth`: Element encoding (torch or pickle format)

**Use Case:**
- Access trained models for inference
- Analyze training dynamics
- Compare different model configurations
- Load predictions for post-analysis

**Example Usage:**
```python
import torch
import pandas as pd

# Load model
model = torch.load('test_training/database_3/test_cn/model.pth')

# Load training history
training_df = pd.read_csv('test_training/database_3/test_cn/training.csv')

# Load predictions
test_df = pd.read_csv('test_training/database_3/test_cn/test_set.csv')

# Load uncertainty quantification
uq_df = pd.read_csv('test_training/database_3/test_cn/uq.csv')
```

---

## Workflow Overview

### 1. **Data Preparation Pipeline**
```
Raw VASP Data (data/DFT_data/)
    ↓
Process Graphs (create graph datasets)
    ↓
test_graph_datasets/ (stores processed graphs)
```

### 2. **Data Loading**
```
Graph Datasets
    ↓
Create DataLoaders
    ↓
DATALOADERS/ (cached for speed)
```

### 3. **Model Training**
```
DataLoaders + Configuration
    ↓
Train Model
    ↓
test_training/ (saves weights and metrics)
```

### 4. **Model Evaluation**
```
Trained Models
    ↓
Nested Cross-Validation (NCV)
    ↓
NCV/ (aggregated results)
```

### 5. **Hyperparameter Optimization**
```
Grid/Random Search of Parameters
    ↓
Train Multiple Models
    ↓
hyperparameter_optimization/ (comparison results)
```

---

## File Size Guidelines

**Typical File Sizes:**

| File Type | Typical Size | Notes |
|-----------|--------------|-------|
| Model weights (`model.pth`) | 5-50 MB | Depends on network size |
| Graph dataset (InMemoryDataset) | 10-500 MB | Depends on dataset size |
| DataLoader (serialized) | 5-20 MB | Compact representation |
| `training.csv` | 1-10 MB | One row per epoch |
| `test_set.csv` | 1-50 MB | One row per test sample |

**Storage Requirements:**
- Single training run: ~100-200 MB
- Full database (multiple runs): 10-100 GB
- All models + datasets: 100 GB - 1 TB

---

## Best Practices

### Organizing New Runs

1. **Use descriptive names:** `ads_height_test_1` (clear purpose)
2. **Include hyperparameters:** `cn_std_3_dim_256` (model config)
3. **Group by database:** `database_3/cn_std_3_dim_256`
4. **Archive old runs:** Move completed experiments to subdirectories

### Loading Models

```python
import torch
import pandas as pd

# Load configuration
with open('test_training/database_3/test_cn/input.txt', 'r') as f:
    config = f.read()

# Load model
model = torch.load('test_training/database_3/test_cn/model.pth')

# Load results
results_df = pd.read_csv('test_training/database_3/test_cn/test_set.csv')
```

### Comparing Multiple Runs

```python
import pandas as pd
import os

# Load all performance summaries
runs = {}
for run_dir in os.listdir('test_training/database_3'):
    csv_path = f'test_training/database_3/{run_dir}/performance.txt'
    if os.path.exists(csv_path):
        runs[run_dir] = pd.read_csv(csv_path)

comparison_df = pd.concat(runs, keys=runs.keys())
print(comparison_df)
```

---

## Related Documentation

- **Data Processing:** See `data/graph_datasets/README.md` for dataset naming convention
- **Training Scripts:** See `scripts/` directory for training code
- **Notebooks:** See `notebooks/` for analysis templates (especially `02_ml_model_analysis.ipynb`)
- **Project README:** See main `README.md` for project overview

---

## Summary

The models directory contains all outputs from the ML pipeline:

- **DATALOADERS/** - Cached data loaders for fast training
- **Experiments/** - Exploratory model runs
- **NCV/** - Rigorous cross-validation results
- **hyperparameter_optimization/** - Hyperparameter tuning results
- **test_graph_datasets/** - Processed graph datasets (input to training)
- **test_training/** - Trained models and predictions (output from training)

Each contains organized results following consistent naming conventions and file structures, enabling reproducibility and easy comparison across experiments.
