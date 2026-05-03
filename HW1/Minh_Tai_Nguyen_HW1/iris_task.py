from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Specify the load_data arguments
# data_path
# mean_subtraction
# normalization
data_path = "iris_dataset.pt"
# Iris feature columns have different scales, so center and normalize before training.
mean_subtraction = True
normalization = True

iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# in_features
# out_size
# hidden_units
# non_linearity
in_features = 4
out_size = 3
# Two tanH hidden layers meet the assignment requirement and exceed 90% accuracy.
hidden_units = [8, 6]
non_linearity = ["tanH", "tanH"]

# create a network based on the architecture
# net
net = create_net(in_features, hidden_units, non_linearity, out_size)

# specify the training opts
# train_opts
train_opts = {
    "num_epochs": 80,
    "lr": 0.01,
    "momentum": 0.9,
    "batch_size": 24,
    "weight_decay": 0.0001,
    "step_size": 80,
    "gamma": 1
}

# Train and save the trained model
train(net, iris_dataset, train_opts)
save(net, "iris_solution.pt")
