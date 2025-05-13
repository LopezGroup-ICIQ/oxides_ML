

# Define multiple hyperparameters and their respective values
hyperparams_to_test = {
    "train.lr0": [1e-6, 1e-5, 1e-4, 1e-3],          # Learning rate
    "train.patience": [6, 5, 4, 3],                 # Patience
    "train.factor": [0.9, 0.8, 0.7, 0.6],           # lr-scheduler scaling factor
    "train.minlr": [1e-6, 1e-7, 1e-8, 1e-9],        # Minimum learning rate 

    "architecture.dim": [128, 64, 32, 16],     # Layer width
    "architecture.num_linear": [0, 1 ,2, 3],        # Number of dense layers at the start of the model
    "architecture.num_conv": [2, 3, 4 ,5],       # Number of convolutional layers
    "architecture.pool_heads": [0, 1 ,2, 3],        # Number of multihead attention blocks in pooling layer
}