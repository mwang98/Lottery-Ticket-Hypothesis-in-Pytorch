import os
import sys
import wandb
import numpy as np

param_ratio = sys.argv[1]
dataset = sys.argv[2]

acc_filepath = f"{os.getcwd()}/dumps/lt/fc1/{dataset}/lt_all_accuracy_{param_ratio}.dat"
loss_filepath = f"{os.getcwd()}/dumps/lt/fc1/{dataset}/lt_all_loss_{param_ratio}.dat"

wandb.init(project="11631-w1", entity="mwang98")
wandb.config.parameter_ratio = param_ratio
wandb.config.dataset = dataset
wandb.run.name = f"{dataset}_{param_ratio}"

acc_data = np.load(acc_filepath, allow_pickle=True)
loss_data = np.load(loss_filepath, allow_pickle=True)

for acc, loss in zip(acc_data, loss_data):
    wandb.log({"Accuracy": acc, "Validation Loss": loss})

last_10_avg_acc = np.mean(acc_data[-10:])
last_10_avg_loss = np.mean(loss_data[-10:])

wandb.log({"Average accuracy of last 10 epochs": last_10_avg_acc,
           "Average validation loss of last 10 epochs": last_10_avg_loss})
