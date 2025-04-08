import os
import toml
import subprocess


# Path to the template input file
input_template = "input.toml"
output_base_dir = "../models/hyperparameter_optimization/initial/Set3"

# Define multiple hyperparameters and their respective values
hyperparams_to_test = {
    "train.lr0": [1e-6, 1e-5, 1e-4, 1e-3],          # Learning rate
    "train.patience": [6, 5, 4, 3, 2],              # Patience
    "train.factor": [0.9, 0.8, 0.7, 0.6, 0.5],      # lr-scheduler scaling factor
    "train.minlr": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9],  # Minimum learning rate 

    "architecture.dim": [256, 128, 64, 32, 16],     # Layer width
    "architecture.num_linear": [0, 1 ,2, 3],        # Number of dense layers at the start of the model
    "architecture.num_conv": [2, 3, 4 ,5, 6],       # Number of convolutional layers
    "architecture.pool_heads": [0, 1 ,2, 3],        # Number of multihead attention blocks in pooling layer
}

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop over different hyperparameters
for hyperparam_name, hyperparam_values in hyperparams_to_test.items():
    print(f"=== Varying hyperparameter: {hyperparam_name} ===")

    # Reload base input file for each new hyperparameter
    with open(input_template, "r") as f:
        base_config = toml.load(f)

    # Loop over all values for the current hyperparameter
    for value in hyperparam_values:
        print(f"Testing {hyperparam_name} = {value}")

        # Create a modified config
        modified_config = base_config.copy()
        
        # Set new hyperparameter value
        keys = hyperparam_name.split(".")  # Handle nested dicts
        sub_dict = modified_config
        for key in keys[:-1]:
            sub_dict = sub_dict[key]
        sub_dict[keys[-1]] = value
        
        # Create a unique input file for this setting
        param_safe_name = hyperparam_name.replace('.', '_')
        value_safe_name = str(value).replace('.', '_')
        input_filename = f"input_{param_safe_name}_{value_safe_name}.toml"
        input_filepath = os.path.join(output_base_dir, f"{param_safe_name}", input_filename)
        
        # Ensure output directory for this hyperparameter exists
        os.makedirs(os.path.dirname(input_filepath), exist_ok=True)

        # Write new config file
        with open(input_filepath, "w") as f:
            toml.dump(modified_config, f)
        
        output_path = os.path.join(output_base_dir, f"{param_safe_name}")
        # Run the training script
        command = f"bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate oxides_ML && python /home/tvanhout/oxides_ML/scripts/train_mve_from_loaders.py -i {input_filepath} -o {output_path}'"
        print(f"Running: {command}")
        
        # Run the command and capture output
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Save logs (append mode)
        log_path = os.path.join(output_base_dir, f"{param_safe_name}", "train_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write(f"\n\n===== Run for {hyperparam_name} = {value} =====\n")
            log_file.write(process.stdout)
            log_file.write("\n--- STDERR ---\n")
            log_file.write(process.stderr)

        # Debugging: Check if training failed
        if process.returncode != 0:
            print(f"Error running: {command}")
            print(process.stderr)

        print(f"Completed run for {hyperparam_name} = {value}")

print("Hyperparameter optimization completed.")

