# Graph Datasets

This directory contains pre-processed PyTorch Geometric (PyG) graph datasets converted from VASP DFT calculations. Each dataset is a serialized `InMemoryDataset` containing graph representations of molecular adsorbates on oxide surfaces.

## Dataset Naming Convention

Each dataset filename follows a strict naming convention that encodes the graph conversion parameters and data selection criteria:

```text
{SOURCE}_{TARGET}_{TOLERANCE}_{SCALING}_{SURFACE_ORDER}_{ADSORBATE}_{RADICAL}_{VALENCE}_{CN}_{MAGNETIZATION}_{ADS_HEIGHT}_{STATE}_{AUGMENT}
```

### Naming Components

| Position | Variable | Type | Description |
|----------|----------|------|-------------|
| 1 | `SOURCE` | str | VASP database identifier (e.g., `database_1`, `database_2`, `database_3`, `database_4`, `database_5`, `database`, `surface_adsorbates`) |
| 2 | `TARGET` | str | Target property being predicted (e.g., `adsorption_energy`, `formation_energy`) |
| 3 | `TOLERANCE` | float (no decimal) | Tolerance in Å for nearest-neighbor cutoff. Remove decimal point (e.g., `0.25` → `025`, `0.3` → `03`, `0.5` → `05`) |
| 4 | `SCALING` | float (no decimal) | Scaling factor for the unit cell. Remove decimal point (e.g., `1.25` → `125`) |
| 5 | `SURFACE_ORDER` | int | Order of nearest neighbors for surface layer identification (e.g., `2`, `3`, `4` or `-1` for custom) |
| 6 | `ADSORBATE` | bool | Include one-hot encoded adsorbate flags as node features (`True`/`False`) |
| 7 | `RADICAL` | bool | Include radical atom indicators as node features (`True`/`False`) |
| 8 | `VALENCE` | bool | Include valence electron information as node features (`True`/`False`) |
| 9 | `CN` | bool/str | Include coordination number (CN) as node feature. Can be `True`, `False`, or `cn` (legacy) |
| 10 | `MAGNETIZATION` | bool | Include magnetization/spin polarization as node features (`True`/`False`) |
| 11 | `ADS_HEIGHT` | bool | Include adsorbate height above surface as node feature (`True`/`False`) |
| 12 | `STATE` | str | Initial structure state used for graph creation (`initial` or `relaxed`) |
| 13 | `AUGMENT` | bool | Data augmentation flag indicating whether both initial and relaxed structures were included (`True`/`False`) |

## Example Dataset Names

### Example 1: Basic Dataset

```text
database_2_adsorption_energy_025_125_2_False_False_False_True_False_False_initial_True
```

**Decoded:**

- Source: `database_2` (VASP database 2)
- Target: `adsorption_energy` (predicting adsorption energy)
- Tolerance: `0.25` Å (nearest-neighbor cutoff)
- Scaling: `1.25` (unit cell scaling factor)
- Surface Order: `2` (2nd-order nearest neighbors for surface identification)
- Node Features: **Disabled** - adsorbate, radical, valence, magnetization, ads_height
- Node Features: **Enabled** - coordination number (CN)
- State: `initial` (using initial VASP structures)
- Augment: `True` (data augmentation with both initial and relaxed structures)

### Example 2: Advanced Dataset with Height Features

```text
database_3_adsorption_energy_025_125_3_False_False_False_True_False_True_initial_True
```

**Decoded:**

- Source: `database_3`
- Target: `adsorption_energy`
- Tolerance: `0.25` Å
- Scaling: `1.25`
- Surface Order: `3` (3rd-order nearest neighbors)
- Node Features: CN (**enabled**), adsorbate height (**enabled**)
- All other features disabled
- State: `initial`
- Augment: `True`

### Example 3: Relaxed Structure Dataset

```text
database_2_adsorption_energy_03_125_2_False_False_False_True_False_False_relaxed_False
```

**Decoded:**

- Source: `database_2`
- Target: `adsorption_energy`
- Tolerance: `0.3` Å (larger cutoff)
- Scaling: `1.25`
- Surface Order: `2`
- Node Features: CN (**enabled**)
- State: `relaxed` (using DFT-relaxed structures)
- Augment: `False` (no augmentation; only relaxed structures)

## Node Features

When enabled in the dataset name, the following node features are included in the graph's `node_feats` list and concatenated to the node feature matrix `x`:

| Feature | Type | Description |
|---------|------|-------------|
| **One-Hot Element Encoding** | float | One-hot encoded chemical element (always included) |
| `ADSORBATE` | binary | Flag indicating if atom is part of the adsorbate (1) or surface (0) |
| `RADICAL` | binary | Indicator of unpaired electrons (for organic molecules) |
| `VALENCE` | int | Number of valence electrons for the atom |
| `CN` | int | Coordination number: number of nearest neighbors within tolerance distance |
| `MAGNETIZATION` | float | Spin magnetic moment (from VASP ISPIN calculations) |
| `ADS_HEIGHT` | float | Height of adsorbate atom above the fitted surface plane (in Å) |

**Note:** The one-hot element encoding is always present regardless of the naming convention. Additional features are appended to `x` in the order shown above when enabled.

## Graph Conversion Parameters

### Tolerance (Å)

Controls the distance cutoff for nearest-neighbor detection and edge creation in the graph.

- **0.25 Å**: Typical for metal oxide surfaces (e.g., RuO2, IrO2, TiO2). Captures first/second nearest neighbors accurately.
- **0.3 Å**: Required for some materials (e.g., Zn-based systems) with slightly larger interatomic distances.
- **0.5 Å**: Looser cutoff for larger unit cells or sparser structures.

### Scaling Factor

Expands or contracts the unit cell before graph construction.

- **1.25**: Standard scaling (25% expansion). Used for most datasets.
- Other values: Experimental scaling for sensitivity analysis or specific material systems.

### Surface Order

Defines which metal atoms are classified as "surface" vs. "bulk" for edge filtering.

- **1**: Only topmost atomic layer
- **2**: 1st and 2nd nearest neighbor metal atoms (most common)
- **3**: Extended to 3rd nearest neighbors for thicker surface regions
- **4**: Very thick surface definition
- **-1**: Custom or experimental surface definition

## Data Organization

Each dataset file contains:

- **Graphs**: Serialized PyG `Data` objects in a collated `InMemoryDataset` format
- **Metadata per graph**:
  - `x` (node features)
  - `edge_index` (edge connectivity)
  - `edge_attr` (edge features, if computed)
  - `y` (target property: adsorption energy)
  - `formula` (molecular formula)
  - `material` (oxide surface: RuO2, IrO2, TiO2)
  - `adsorbate_name` (molecule name)
  - `state` (initial or relaxed from DFT)
  - `ads_energy` (adsorption energy in eV)
  - `graph_id` (unique graph identifier)

## Loading a Dataset

To load and use a dataset in your code:

```python
from src.oxides_ml.classes import OxidesGraphDataset

# Define graph conversion parameters
graph_params = {
    "structure": {
        "tolerance": 0.25,
        "scaling_factor": 1.25,
        "surface_order": 2
    },
    "features": {
        "adsorbate": False,
        "radical": False,
        "valence": False,
        "cn": True,              # Coordination number enabled
        "magnetization": False,
        "ads_height": False
    },
    "target": "adsorption_energy"
}

# Load the dataset
dataset = OxidesGraphDataset(
    vasp_directory="/path/to/VASP/data",
    graph_dataset_dir="/path/to/graph_datasets",
    graph_params=graph_params,
    initial_state=True,
    augment=True,
    force_reload=False
)

# Access individual graphs
graph = dataset[0]
print(f"Formula: {graph.formula}")
print(f"Adsorption Energy: {graph.ads_energy.item()} eV")
print(f"Node features: {graph.node_feats}")
print(f"Number of nodes: {graph.num_nodes}")
print(f"Number of edges: {graph.num_edges}")
```

## Dataset Variants

The repository contains multiple datasets with different configurations:

### By Source Database

- `database_1_*`: First experimental/reference dataset
- `database_2_*`: Secondary dataset with additional materials/structures
- `database_3_*`: Large-scale curated dataset (commonly used)
- `database_4_*`: Extended dataset variant
- `database_5_*`: Additional high-quality subset
- `database_*`: Combined/aggregated dataset
- `surface_adsorbates_*`: Surface-adsorbate specific dataset

### By Augmentation Status

- `*_initial_True`: Data augmentation enabled; includes both initial and relaxed structures
- `*_initial_False`: Single state datasets (only initial or only relaxed)
- `*_relaxed_True`: Augmented with relaxed structures
- `*_relaxed_False`: Only relaxed structures, no augmentation

### By Feature Set

- Minimal: `False_False_False_False_False_False` (only element one-hot encoding)
- Coordination: `False_False_False_True_False_False` (+ coordination number)
- Extended: `False_False_False_True_False_True` (+ coordination number + adsorbate height)

## Cache Files

When creating datasets, the following cache files are generated in the notebook/script directory:

- **`adsorbate_indices_cache.json`**: Maps adsorbate molecule names to atom indices for faster loading
- **`path_to_id_cache.json`**: Maps file paths to unique graph IDs for traceability

These caches improve performance during dataset creation and enable reproducibility.

## Notes

- Graph datasets are **immutable** once created; they are serialized PyG `InMemoryDataset` objects
- Tolerance values are stored without decimals in the filename (e.g., `025` for `0.25`, `03` for `0.3`)
- The augmentation flag indicates whether the dataset builder considered both initial and relaxed structures during graph generation
- Each graph includes metadata such as material composition, adsorbate name, and calculated properties from DFT
