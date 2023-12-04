# %%
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
from tensorflow.keras.layers import *
import tensorflow as tf
import keras
from keras import Model
from keras import regularizers
from keras.layers import Dense, Dropout
from keras import layers
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import pathlib
import ast
import json
import coral_ordinal as coral
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from imblearn.over_sampling import SMOTENC
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure
import math
import numpy as np
import math
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure
sns.set(rc={'figure.figsize':(9, 7)})
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Coral Ordinal Function and Custom metric defining here
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MODEL_PATH = '../data/models/'
DATA_DIR = '../data/'


# %%
# Important functions for test accuracy and confusion matrix
def predict_get_f1_accuracy(y_true, y_pred, num_classes):
    # Resetting the index
    y_true = y_true.reset_index(drop=True)

    # Function for custom distance accuracy metric
    def accuracy_metric(y_true, y_pred):
        a = []

        for i in range(len(y_true)):
            yt = y_true[i]
            yp = y_pred[i]

            temp = 1 - (np.abs(yt - yp) / num_classes)
            a.append(temp)
        # print(a)
        accuracy = sum(a) / len(y_true)

        return accuracy

    # Getting the f1 score using sklearn.metrics.precision_recall_fscore_support
    f1_score_test = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Getting the custom distance accuracy metric
    custom_accuracy_test = accuracy_metric(y_true, y_pred)

    # Returing the values
    return f1_score_test[2], custom_accuracy_test


def save_confusion_matrix_func(y_test, yp, title, path, route_id):
    cm = confusion_matrix(y_test, yp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = disp.plot(ax=ax)
    # Add a title to the plot
    ax.set_title(title + " - Real Count")
    plt.savefig(f'{path}/CM_real_count_{route_id}.jpg')

    # Getting the percentage CM
    cm = ((cm * 100) / (cm.sum(axis=1)[:, np.newaxis]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    _ = disp.plot(ax=ax)
    # Add a title to the plot
    ax.set_title(title + " - Percentages")
    plt.savefig(f'{path}/CM_percentage_{route_id}.jpg')



DATA_DIR = '../data'

pd.set_option('display.max_columns', None)

import argparse
# Create the parser
parser = argparse.ArgumentParser(description="Script for generating the deep ensemble hyperparameter results")
# Add the --value argument
parser.add_argument('--dv', type=int, help="Data version")
# Parse the arguments
args = parser.parse_args()
# Asking the user for their name
data_type = int(args.dv)

path = f'../data/new_data/data{data_type}/'
train_df = pd.read_csv(f'{path}/Data{data_type}_Train.csv')
val_df = pd.read_csv(f'{path}/Data{data_type}_Val.csv')
test_df = pd.read_csv(f'{path}/Data{data_type}_Test.csv')

tensor_train_data = torch.Tensor(train_df.Input1).unsqueeze(1)
tensor_train_label = torch.Tensor(train_df.Output)

tensor_val_data = torch.Tensor(val_df.Input1).unsqueeze(1)
tensor_val_label = torch.Tensor(val_df.Output)

tensor_test_data = torch.Tensor(test_df.Input1).unsqueeze(1)
tensor_test_label = torch.Tensor(test_df.Output)

train_dataset = TensorDataset(tensor_train_data, tensor_train_label)
val_dataset = TensorDataset(tensor_val_data, tensor_val_label)


import torch
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler

class MCDO(nn.Module):
    def __init__(self, num_features, hidden_size, n_layers, drop_out):
        super(MCDO, self).__init__()

        layers = []
        input_size = num_features

        # Create 'n' linear layers
        for _ in range(n_layers - 1):  # n_layers - 1 because the last layer is the output layer
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size

        # The last layer outputs a single value
        self.out = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(torch.relu(layer(x)))
        out = self.out(x)  # Apply the output layer
        return out



# %%
models = {}

# %%
def train_mcdo(config):

    hidden_size = int(config['hidden_size'])
    num_layers = int(config['num_layers'])
    patience = int(config['patience'])
    batch_size = int(config['batch_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model_name = f'hs_{hidden_size}_nl_{num_layers}_do_{config["drop_out"]}_ep_{config["epochs"]}_p_{config["patience"]}_lr_{config["lr"]}_bs_{batch_size}'

    models[f'{model_name}']= MCDO(config["num_features"], hidden_size, num_layers, config["drop_out"])

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(models[f'{model_name}'].parameters(), lr=config["lr"])
    device = 'cpu'
    models[f'{model_name}'].to(device)
    
    best_model_dir = f'./models/'
    pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True) 
    best_model_path = f'{best_model_dir}/hs_{hidden_size}_nl_{num_layers}_do_{config["drop_out"]}_ep_{config["epochs"]}_p_{config["patience"]}_lr_{config["lr"]}_bs_{batch_size}.pt'
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        models[f'{model_name}'].train()
        # Train loop
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            out = models[f'{model_name}'](x)
            loss = criterion(y, out)
            loss.backward()
            optimizer.step()

        # Validation loop
        models[f'{model_name}'].eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                out = models[f'{model_name}'](x)
                val_loss = criterion(y, out)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        # Early stopping and model saving logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the model state
            torch.save(models[f'{model_name}'].state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break


    tune.report(train_loss=loss,best_val_loss=best_val_loss)

# %%


def main(num_samples=100, max_num_epochs=500):
    

    ray.init()

    config = {
        "num_features": 1,
        "hidden_size": tune.quniform(1, 200, 1),
        "num_layers": tune.quniform(2, 10, 1),
        "drop_out": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 1e-1),
        "epochs": max_num_epochs,
        "patience": tune.quniform(1, 50, 1),
        "batch_size": tune.quniform(1, 20, 1)
    }

    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.bayesopt import BayesOptSearch
    algo = BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0})
    algo = ConcurrencyLimiter(algo, max_concurrent=50)

    result = tune.run(
            tune.with_parameters(train_mcdo),
            config=config,
            resources_per_trial={"cpu": 1},
            metric="best_val_loss",
            mode="min",
            search_alg=algo,
            num_samples=num_samples,
            # max_concurrent_trials=20,
            local_dir= f'/home/sgupta/WORK/Triplevel_transformer_model/baselines/hyperparameter_search',
            name=f"experiment_mcdo_data2",
            max_failures=7,
            raise_on_failed_trial=False
        )

    temp = result.dataframe()   
    hyperparams_path = f'./hyperparameter_search/'
    temp.to_csv(f"{hyperparams_path}/RAY_RESULTS_mcdo_data{data_type}.csv")


# %%
import ray

main(100,500)
#  Stop Ray
ray.shutdown()

# %%
models


