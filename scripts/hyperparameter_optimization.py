import os
import toml
import argparse
import subprocess
from pathlib import Path

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Run GNN training for multiple dataloader configs.")
parser.add_argument("-d", "--database", type=str, required=True, help="Name of the database folder in ../models/DATALOADERS/")
parser.add_argument("-t", "--template", type=str, required=True, help="Path to base .toml input config file.")
parser.add_argument("--train_script", type=str, default="train_mve.py", help="Path to training script.")
args = parser.parse_args()

# --- Paths ---
dataloader_root = Path("../models/DATALOADERS") / args.database
input_template = Path(args.template)
train_script = Path(args.train_script).resolve()

if not train_script.exists():
    raise FileNotFoundError(f"Training script not found: {train_script}")

hyperopt_output_root = Path("../models/hyperparameter_optimization")
hyperopt_output_root.mkdir(parents=True, exist_ok=True)

# --- Define hyperparameters to sweep (can be left empty) ---
hyperparams_to_test = {
    # Example:
    # "train.lr0": [1e-6, 1e-5],
}

# --- Number of runs per split ---
n_runs = 5  # Hardcoded

# --- Build dataloader configs ---
all_dataloader_configs = []
for set_dir in dataloader_root.glob("*"):
    if set_dir.is_dir():
        config = {
            "name": f"{args.database}_{set_dir.name}",
            "loader_dir": str(set_dir.resolve()),
        }
        all_dataloader_configs.append(config)

# --- Loop over each dataloader config ---
for config in all_dataloader_configs:
    print(f"\n=== Running config: {config['name']} ===")

    if hyperparams_to_test:
        for param, values in hyperparams_to_test.items():
            for val in values:
                for run_id in range(1, n_runs + 1):
                    with open(input_template, "r") as f:
                        base_config = toml.load(f)

                    # Apply hyperparameter change
                    keys = param.split(".")
                    sub_config = base_config
                    for k in keys[:-1]:
                        sub_config = sub_config[k]
                    sub_config[keys[-1]] = val

                    param_name = param.split(".")[-1]
                    param_key = param.replace(".", "-")
                    val_str = str(val).replace(".", "p") if isinstance(val, float) else str(val)

                    output_dir = (
                        hyperopt_output_root
                        / args.database
                        / f"run_{run_id}"
                        / Path(config["loader_dir"]).name
                        / param_name
                        / f"{param_key}_{val_str}"
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)

                    config_path = output_dir / "input_config.toml"
                    with open(config_path, "w") as f:
                        toml.dump(base_config, f)

                    env = os.environ.copy()

                    log_file = output_dir / "log.txt"
                    with open(log_file, "w") as f:
                        subprocess.run([
                            "python", str(train_script),
                            "-i", str(config_path),
                            "-o", str(output_dir),
                            "--loader_dir", config["loader_dir"]
                        ], env=env, stdout=f, stderr=subprocess.STDOUT)

    else:
        # No hyperparameter sweep: just run n_runs on base config
        for run_id in range(1, n_runs + 1):
            with open(input_template, "r") as f:
                base_config = toml.load(f)

            output_dir = (
                hyperopt_output_root
                / args.database
                / f"run_{run_id}"
                / Path(config["loader_dir"]).name
                / "base"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            config_path = output_dir / "input_config.toml"
            with open(config_path, "w") as f:
                toml.dump(base_config, f)

            env = os.environ.copy()

            log_file = output_dir / "log.txt"
            with open(log_file, "w") as f:
                subprocess.run([
                    "python", str(train_script),
                    "-i", str(config_path),
                    "-o", str(output_dir),
                    "--loader_dir", config["loader_dir"]
                ], env=env, stdout=f, stderr=subprocess.STDOUT)

print("Hyperparameter optimization completed.")
