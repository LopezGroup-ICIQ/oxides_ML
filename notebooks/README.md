# Notebooks Directory

This directory contains Jupyter notebooks for dataset analysis, model evaluation, and utilities for the GAME-Net-Ox project.

## Quick Start: Environment Variables

All notebooks in this directory have been generalized to use **environment variables** with sensible defaults. This makes notebooks portable across different machines and setups.

### Setting Environment Variables

**Option 1: Temporary (current shell only)**
```bash
export VASP_DATA_DIR="/path/to/your/VASP/data"
export GRAPH_DATASET_DIR="./models/test_graph_datasets"
export TRAINING_RESULTS_DIR="./models/test_training"
jupyter lab
```

**Option 2: Permanent (in ~/.bashrc or ~/.bash_profile)**
```bash
echo 'export VASP_DATA_DIR="/path/to/your/VASP/data"' >> ~/.bashrc
source ~/.bashrc
```

**Option 3: Using `.env` file (if supported by your IDE)**
Create a `.env` file in the project root:
```
VASP_DATA_DIR=/path/to/your/VASP/data
GRAPH_DATASET_DIR=./models/test_graph_datasets
TRAINING_RESULTS_DIR=./models/test_training
```

### Available Environment Variables

| Variable | Notebook(s) | Default | Description |
|----------|-----------|---------|-------------|
| `VASP_DATA_DIR` | 01_*, Graph*, DFT_* | `/path/to/VASP/data` | Path to VASP DFT data |
| `GRAPH_DATASET_DIR` | 01_*, Graph*, Create_* | `./models/test_graph_datasets` | Output directory for graph datasets |
| `TRAINING_RESULTS_DIR` | 02_*, ML_* | `./models/test_training` | Path to training results |
| `MODEL_DIRECTORY` | 02_*, ML_*, Generate_* | `database_3/test_cn` | Relative path to specific model |
| `DFT_DATA_CSV` | DFT_* | `./data/VASP_dataset.csv` | Path to DFT dataset CSV |
| `VASP_ADSORBATE_EXAMPLE` | Adsorbate_* | `/path/to/example/CONTCAR` | Example VASP file for testing |
| `DATALOADERS_OUTPUT_DIR` | Create_* | `./models/DATALOADERS/RELAXED` | Output directory for dataloaders |
| `EXPERIMENTS_DIR` | Generate_* | `./models/Experiments/RELAXED/tolerance_fixed/` | Base experiments directory |
| `EXPERIMENT_NAME` | Generate_* | `Db1_TiO2_base` | Name of specific experiment |
| `GAS_PHASE_DATA_DIR` | Gas_* | `/path/to/database/gas_phase` | Path to gas phase VASP data |
| `HYPERPARAMETER_OPT_DIR` | ML_comparison (initial) | `./models/hyperparameter_optimization/initial_test/initial` | Initial hyperparameter optimization |
| `HYPERPARAMETER_OPT_DIR_AUGMENT` | ML_comparison (augment) | `./models/hyperparameter_optimization/initial_test/augment` | Augmented hyperparameter optimization |
| `HYPERPARAMETER_OPT_BASE_DIR` | ML_comparison (function) | `./models/hyperparameter_optimization` | Base hyperparameter optimization |

### Example Workflow

```bash
# Set your paths
export VASP_DATA_DIR="/BACKUP/database_3/oxide_adsorbates"
export GRAPH_DATASET_DIR="/home/user/oxides_ML/models/test_graph_datasets"

# Start Jupyter
jupyter lab notebooks/

# Open 01_graph_dataset_analysis.ipynb
# → Automatically uses your paths from environment variables
# → All defaults are relative paths for portability
```

## Overview

The notebooks have been reorganized and generalized to provide reusable templates for common analysis tasks. Notebooks are numbered sequentially by purpose:

- `01_*`: Graph dataset analysis
- `02_*`: ML model analysis and evaluation
- `03_*`: Utility functions and helpers

## Generalized Notebooks

These are the **primary analysis notebooks**. Each is designed to be reusable with configurable parameters at the top of the notebook.

### `01_graph_dataset_analysis.ipynb`

**Purpose:** Analyze PyTorch Geometric graph datasets with configurable graph construction parameters.

**Configuration Parameters:**

- `VASP_DIRECTORY`: Path to VASP data directory
- `GRAPH_DATASET_DIR`: Output directory for processed graphs
- `TOLERANCE`: Nearest-neighbor cutoff distance (Å)
- `SCALING_FACTOR`: Unit cell scaling factor
- `SURFACE_ORDER`: Order of nearest neighbors for surface classification
- Node feature flags (coordination number, magnetization, adsorption height, etc.)
- `TARGET_PROPERTY`: Property to predict (default: `adsorption_energy`)
- `USE_INITIAL_STATE`: Use initial (True) or relaxed (False) structures
- `USE_AUGMENTATION`: Include both initial and relaxed structures

**Features:**

- Dataset loading with configurable parameters
- Graph inspection and statistics
- Material and type distribution analysis
- Target property statistics and visualization
- Graph filtering and search by criteria
- Adsorbate distribution analysis

**Usage:**

1. Open the notebook
2. Modify the configuration parameters in the "Configuration" section
3. Run cells sequentially to analyze the dataset

**Example Configurations:**

```python
# Configuration A: Basic analysis with coordination number
TOLERANCE = 0.25
SCALING_FACTOR = 1.25
SURFACE_ORDER = 2
INCLUDE_CN = True  # Coordination number enabled

# Configuration B: Full features analysis
TOLERANCE = 0.3
SCALING_FACTOR = 1.25
SURFACE_ORDER = 3
INCLUDE_CN = True
INCLUDE_ADS_HEIGHT = True
```

---

### `02_ml_model_analysis.ipynb`

**Purpose:** Analyze trained ML model performance, learning curves, and prediction accuracy.

**Configuration Parameters:**

- `BASE_PATH`: Base directory for training results
- `MODEL_DIRECTORY`: Relative path to model output (e.g., `database_3/test_cn`)
- `DATASETS_TO_LOAD`: Dictionary of CSV files to load (training, train_set, test_set, validation_set, uncertainty)

**Features:**

- Automatic dataset loading from training outputs
- Learning curve visualization (MAE, loss, R² score)
- Prediction vs. true value scatter plots
- Error analysis and distribution
- Per-atom and per-material performance evaluation
- Uncertainty quantification analysis (if available)

**Usage:**

1. Open the notebook
2. Set `BASE_PATH` to your training results directory
3. Set `MODEL_DIRECTORY` to your specific model run
4. Run cells sequentially

**Example Configurations:**

```python
# Configuration A: Analyze database_3 model
MODEL_DIRECTORY = "database_3/test_cn"

# Configuration B: Analyze database_2 model
MODEL_DIRECTORY = "database_2/surface_order_2_cn"

# Configuration C: Segmented model evaluation
MODEL_DIRECTORY = "database_3/segmented_test"
```

---

### `03_dataset_utilities.ipynb`

**Purpose:** Reusable utility functions for dataset processing, coordinate transformations, and atomic calculations.

**Includes:**

- **Adsorption Height Functions:**
  - `get_adsorption_heights_per_atom()`: Height for each adsorbate atom
  - `get_adsorption_heights_average()`: Mean adsorption height
  - `get_adsorption_heights_min()`: Closest adsorbate-surface distance

- **Coordinate Functions:**
  - `read_poscar_coordinates()`: Read VASP POSCAR/CONTCAR files
  - `cart_to_direct()`: Convert between coordinate systems

- **Atomic Index Functions:**
  - `get_element_indices()`: Find atoms of specific element
  - `get_element_indices_multiple()`: Batch element queries
  - `get_surface_atoms()`: Identify surface metal atoms
  - `compute_distances()`: Pairwise distances with PBC
  - `get_nearest_neighbors()`: K-nearest neighbors search

**Usage:**

- Import functions from this notebook into your own analysis notebooks
- Use as a library for dataset preprocessing and analysis tasks

---

## Specialized Analysis Notebooks

These notebooks are designed for specific tasks and should be used as-is or modified for specific analysis:

- **`Adsorbate_indices.ipynb`**: Utility for determining adsorbate atom indices
- **`Create_fixed_dataloaders.ipynb`**: Dataset loader creation and configuration
- **`DFT_data_analysis.ipynb`**: Large-scale DFT dataset analysis and visualization
- **`Gas_phase_check.ipynb`**: Validation of gas-phase molecule calculations
- **`Generate_plots.ipynb`**: High-quality plot generation for papers/reports
- **`GraphDataset_test.ipynb`**: Development and testing of graph dataset creation
- **`Graph_data_analysis.ipynb`**: Specific graph dataset analysis (use `01_graph_dataset_analysis.ipynb` instead for new projects)
- **`ML_comparison.ipynb`**: Comparison of multiple ML models
- **`ML_data_analysis.ipynb`**: Specific model analysis (use `02_ml_model_analysis.ipynb` instead for new projects)

## Workflow Recommendations

### For Dataset Analysis

1. Use `01_graph_dataset_analysis.ipynb` with your dataset parameters
2. For custom processing, reference `03_dataset_utilities.ipynb` for helper functions
3. For detailed DFT analysis, consult `DFT_data_analysis.ipynb`

### For Model Evaluation

1. Use `02_ml_model_analysis.ipynb` with your model directory
2. For model comparison across multiple runs, reference `ML_comparison.ipynb`
3. For publication-quality plots, use `Generate_plots.ipynb`

### For Development

1. Use `GraphDataset_test.ipynb` for dataset creation development
2. Use `03_dataset_utilities.ipynb` as a utility library
3. Use `tmp.ipynb` alternative functions from `03_dataset_utilities.ipynb`

---

## Dependencies

Ensure the following packages are installed:

```bash
pip install torch torch-geometric pandas seaborn matplotlib plotly numpy scipy scikit-learn
```

For GPU support, install PyTorch with CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

See `requirements.txt` and `environment.yml` in the project root for full dependency specifications.

---

## Notes

- **Cell execution:** Run cells sequentially from top to bottom
- **Path configuration:** Update paths to match your system/data locations
- **Reproducibility:** All parameters are configurable for full reproducibility
- **Saving outputs:** Add cells to save plots and results if needed
- **Customization:** Feel free to modify and extend these notebooks for your analysis needs
