import os
import toml
import subprocess


# Path to the template input file
input_template = "input.toml"
output_base_dir = "../models/hyperparameter_optimization/Set1/test_learning_rate"

# Define the hyperparameter to vary
hyperparam_name = "train.lr0"  # Example: Learning rate
hyperparam_values = [1e-2, 1e-3]  # Values to test

# Load the base input file
with open(input_template, "r") as f:
    base_config = toml.load(f)

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Loop over hyperparameter values
for value in hyperparam_values:
    # Create a modified config
    modified_config = base_config.copy()
    
    # Set new hyperparameter value
    keys = hyperparam_name.split(".")  # Handle nested dicts
    sub_dict = modified_config
    for key in keys[:-1]:
        sub_dict = sub_dict[key]
    sub_dict[keys[-1]] = value
    
    # Create a unique input file for this setting
    input_filename = f"input_{hyperparam_name.replace('.', '_')}_{str(value).replace('.', '_')}.toml"
    input_filepath = os.path.join(output_base_dir, input_filename)
    
    with open(input_filepath, "w") as f:
        toml.dump(modified_config, f)
    

    # Run the training script
    command = f"python train_mve.py -i {input_filepath} -o {output_base_dir}"
    print(f"Running: {command}")
    
    # Run the command and capture output
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Save logs
    with open(os.path.join(output_base_dir, "train_log.txt"), "w") as log_file:
        log_file.write(process.stdout)
        log_file.write("\n--- STDERR ---\n")
        log_file.write(process.stderr)

    print(f"Completed run for {hyperparam_name} = {value}")

print("Hyperparameter optimization completed.")
