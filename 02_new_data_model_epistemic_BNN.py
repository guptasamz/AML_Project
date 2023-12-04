
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
from tensorflow.keras.layers import *
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure
import math
import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure
import argparse
import torch.nn as nn
import torch.nn.functional as F

sns.set(rc={'figure.figsize':(9, 7)})
figure(figsize=(9, 7))
pd.set_option('display.max_columns', None)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
MODEL_PATH = '../data/models/'
DATA_DIR = '../data/'
n = 300

# Create the parser
parser = argparse.ArgumentParser(description="Script for generating the results")
# Add the --value argument
parser.add_argument('--dv', type=int, help="Data version")
# Parse the arguments
args = parser.parse_args()
# Asking the user for their name
data_type = int(args.dv)
num_features = 1

path = f'../data/new_data/data{data_type}/'

train_df = pd.read_csv(f'{path}/Data{data_type}_Train.csv')
val_df = pd.read_csv(f'{path}/Data{data_type}_Val.csv')
test_df = pd.read_csv(f'{path}/Data{data_type}_Test.csv')

data_plt_op_dir = f'./results/data{data_type}'
pathlib.Path(data_plt_op_dir).mkdir(parents=True, exist_ok=True) 

scatter(train_df.Input1, train_df.Output, c="red", marker="*")
output_path = f'{data_plt_op_dir}/data_train_{data_type}.png'
plt.savefig(output_path)
plt.clf()

scatter(test_df.Input1, test_df.Output, c="red", marker="*")

output_path = f'{data_plt_op_dir}/data_test_{data_type}.png'
plt.savefig(output_path)
plt.clf()

def get_mcdo_hyperparameters(data_type):
    hp_search_data_path =f'./hyperparameter_search/RAY_RESULTS_mcdo_data{data_type}.csv'
    hp_df = pd.read_csv(hp_search_data_path)

    hp_df = hp_df.sort_values('best_val_loss')
    hp_df.columns = hp_df.columns.str.replace('config/', '')

    return int(hp_df['batch_size'].iloc[0]), int(hp_df['hidden_size'].iloc[0]), int(hp_df['num_layers'].iloc[0]), hp_df['drop_out'].iloc[0], hp_df['lr'].iloc[0], int(hp_df['patience'].iloc[0]), hp_df['epochs'].iloc[0]

batch_size, hidden_size, num_layers, drop_out, learning_rate, patience, epochs =get_mcdo_hyperparameters(data_type)
print(f'MCDO hyperparameters for data set: {data_type}')
print(batch_size, hidden_size, num_layers, drop_out, learning_rate, patience, epochs)

tensor_train_data = torch.Tensor(train_df.Input1).unsqueeze(1)
tensor_train_label = torch.Tensor(train_df.Output)

tensor_val_data = torch.Tensor(val_df.Input1).unsqueeze(1)
tensor_val_label = torch.Tensor(val_df.Output)

tensor_test_data = torch.Tensor(test_df.Input1).unsqueeze(1)
tensor_test_label = torch.Tensor(test_df.Output)


def normalize_values(x):
    max_x = torch.max(x,dim=0)
    min_x = torch.min(x,dim=0)
    range_x = max_x.values - min_x.values
    #Normalizing
    x = (x - min_x.values)/range_x

    return x


train_dataset = TensorDataset(tensor_train_data, tensor_train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(tensor_val_data, tensor_val_label)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(tensor_test_data, tensor_test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

############### Monte Carlo Dropout ##########################

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

model = MCDO(num_features, hidden_size, num_layers, drop_out)
print(model)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

train_df.Input1.min()
train_df.Input1.max()


def make_plot_mcdo(model, data_type, epoch, samples = 50):
    model.train()
    preds = [model(tensor_test_data) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0).detach().numpy()
    stds =  preds.std(axis=0).detach().numpy()
    dfs = []
    y_vals = [means, means+2*stds, means-2*stds]

    for i in range(3):
      data = {
            "x": list(tensor_test_data.squeeze().numpy()),
            "y": list(y_vals[i].squeeze())
      }
      temp = pd.DataFrame.from_dict(data)
      dfs.append(temp)

    df = pd.concat(dfs).reset_index()

    # Plot predictions
    sns_plot = sns.lineplot(data=df, x="x", y="y")

    start = train_df.Input1.min()
    end = train_df.Input1.max()

    print(start)
    # Getting range
    plt.axvline(x=start)
    plt.axvline(x=end)

    # Plot data on top of the uncertainty region
    scatter(train_df.Input1, train_df.Output, c="green", marker="*", alpha=0.1)

    op_dir = f'./results/data{data_type}'
    pathlib.Path(op_dir).mkdir(parents=True, exist_ok=True) 

    output_path = f'{op_dir}/data{data_type}_mcdo_epoch{epoch}.png'
    plt.savefig(output_path)
    plt.clf()
    plt.show()


criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = 'cpu'
model.to(device)

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = f'./model_data/mcdo/best_model_mcdo_data_type_{data_type}.pt'

for epoch in range(epochs):
    model.train()
    # Train loop
    for batch in train_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(y, out)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            out = model(x)
            val_loss = criterion(y, out)
            val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    # Early stopping and model saving logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save the model state
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    print(f'Epoch: {epoch}|Train Loss: {loss}|Val Loss: {avg_val_loss}')

    if epoch % 10 == 0:
        all_test_losses = []
        # Testing
        for batch in test_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)

                # Sampling monte carlo Dropout predictions - 10 samples 
                outs = []
                for i in range(10):
                    out = model(x)
                    outs.append(out)

                # Taking the mean of the prediction 
                out = sum(outs)/len(outs)
                all_test_losses.append(criterion(y, out).item())
        test_loss = sum(all_test_losses)/len(all_test_losses)
        print(f"Epoch {epoch} | batch train loss: {loss} | test loss: {test_loss}")
        make_plot_mcdo(model, data_type, epoch,50)

# After training, load the best model   
model.load_state_dict(torch.load(best_model_path))  

# Getting the prediction for the last epoch
if epoch-1 % 10 != 0:
        all_test_losses = []
        # Test loop
        for batch in test_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)

                # Sampling monte carlo Dropout predictions - 10 samples 
                outs = []
                for i in range(10):
                    out = model(x)
                    outs.append(out)

                # Taking the mean of the prediction 
                out = sum(outs)/len(outs)
                all_test_losses.append(criterion(y, out).item())
        test_loss = sum(all_test_losses)/len(all_test_losses)
        print(f"Epoch {epoch} | batch train loss: {loss} | test loss: {test_loss}")
        make_plot_mcdo(model,data_type, epoch,50)


################# Copleted MCDO model  ##############################
#####################################################################

################# Deep Ensemble Model ##############################

def get_de_hyperparameters(data_type):
    hp_search_data_path =f'./hyperparameter_search/RAY_RESULTS_de_data{data_type}.csv'
    hp_df = pd.read_csv(hp_search_data_path)

    hp_df = hp_df.sort_values('best_val_loss')
    hp_df.columns = hp_df.columns.str.replace('config/', '')

    return int(hp_df['batch_size'].iloc[0]), int(hp_df['hidden_size'].iloc[0]), int(hp_df['num_layers'].iloc[0]), hp_df['lr'].iloc[0], int(hp_df['patience'].iloc[0]), hp_df['epochs'].iloc[0]

batch_size, hidden_size, num_layers, learning_rate, patience, epochs = get_de_hyperparameters(data_type)
print(f'Deep ensemble hyperparameters for data set: {data_type}')
print(batch_size, hidden_size, num_layers, learning_rate, patience, epochs)
num_models =  5


train_dataset = TensorDataset(tensor_train_data, tensor_train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(tensor_val_data, tensor_val_label)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(tensor_test_data, tensor_test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# %%
class DeepEnsembleNet(nn.Module):
    def __init__(self, num_features, hidden_size, n_layers):
        super(DeepEnsembleNet, self).__init__()

        layers = []
        input_size = num_features

        # Create 'n' hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size  # The output of each layer is the input to the next

        self.layers = nn.ModuleList(layers)
        self.mu = nn.Linear(hidden_size, 1)
        self.var = nn.Linear(hidden_size, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        mu = self.mu(x)
        var = torch.exp(self.var(x))
        return mu, var

model = DeepEnsembleNet(num_features,hidden_size, num_layers)
print(model)
print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

def make_plot_de(model,data_type,epoch):
    mus = []
    vars = []
    for m in model:
        mu, var = m(tensor_test_data)
        mus.append(mu)
        vars.append(var)

    means = torch.stack(mus).mean(axis=0).detach().numpy()
    stds = torch.stack(mus).std(axis=0).detach().numpy()**(1/2)

    dfs = []
    y_vals = [means, means+2*stds, means-2*stds]

    for i in range(3):
      data = {
            "x": list(tensor_test_data.squeeze().numpy()),
            "y": list(y_vals[i].squeeze())
      }
      temp = pd.DataFrame.from_dict(data)
      dfs.append(temp)

    df = pd.concat(dfs).reset_index()

    # Plot predictions with confidence
    sns_plot = sns.lineplot(data=df, x="x", y="y")

    start = train_df.Input1.min()
    end = train_df.Input1.max()

    # Highlight training range
    plt.axvline(x=start)
    plt.axvline(x=end)

    # Plot data on top of the uncertainty region
    scatter(train_df.Input1, train_df.Output, c="green", marker="*", alpha=0.1)

    op_dir = f'./results/data{data_type}'
    pathlib.Path(op_dir).mkdir(parents=True, exist_ok=True) 

    output_path = f'{op_dir}/data{data_type}_de_epoch{epoch}.png'
    plt.savefig(output_path)
    plt.show()
    plt.clf()

# Construct ensemble
# num_models = 5
deep_ensemble = [DeepEnsembleNet(num_features,hidden_size, num_layers).to(device) for i in range(num_models)]
criterion = torch.nn.GaussianNLLLoss(eps=1e-02)
optimizers = [optim.Adam(m.parameters(), lr=learning_rate) for m in deep_ensemble]

# Early stopping parameters
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_path = f'./model_data/deep_ensemble/best_model_de_data_type_{data_type}.pt'

for epoch in range(epochs):
    # Train loop
    for batch in train_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        losses = []
        mus = []
        vars = []
        for i, model in enumerate(deep_ensemble):
            optimizers[i].zero_grad()
            mu, var = model(x)
            loss = criterion(mu, y, var)
            loss.backward()
            optimizers[i].step()

            losses.append(loss.item())
            mus.append(mu)
            vars.append(var)
        loss = sum(losses)/len(losses)

    # Validation loop to monitor early stopping
    model.eval()  # Set the model to evaluation mode
    val_losses = []
    for batch in val_loader:  # Assuming you have a validation data loader
        x = batch[0].to(device)
        y = batch[1].to(device)

        with torch.no_grad():  # No gradient calculation for validation data
            for model in deep_ensemble:
                mu, var = model(x)
                val_loss = criterion(mu, y, var)
                val_losses.append(val_loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)

    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break

    print(f'Epoch: {epoch}|Train Loss: {loss}|Val Loss: {avg_val_loss}')

    if epoch % 10 == 0:
        all_test_losses = []
        # Test loop
        for batch in test_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)

            test_losses = []
            mus = []
            vars = []
            for i, model in enumerate(deep_ensemble):
                optimizers[i].zero_grad()
                mu, var = model(x)
                test_loss = criterion(mu, y, var)
                optimizers[i].step()

                test_losses.append(test_loss.item())
                mus.append(mu)
                vars.append(var)
            all_test_losses.append((sum(test_losses)/len(test_losses)))
        test_loss = sum(all_test_losses)/len(all_test_losses)
        print(f"Epoch {epoch} | batch train loss: {loss} | test loss: {test_loss}")
        make_plot_de(deep_ensemble,data_type,epoch)

# After training, load the best model
model.load_state_dict(torch.load(best_model_path))

if epoch-1 % 10 != 0:
    all_test_losses = []
    # Test loop
    for batch in test_loader:
        x = batch[0].to(device)
        y = batch[1].to(device)

        test_losses = []
        mus = []
        vars = []
        for i, model in enumerate(deep_ensemble):
            optimizers[i].zero_grad()
            mu, var = model(x)
            test_loss = criterion(mu, y, var)
            optimizers[i].step()

            test_losses.append(test_loss.item())
            mus.append(mu)
            vars.append(var)
        all_test_losses.append((sum(test_losses)/len(test_losses)))
    test_loss = sum(all_test_losses)/len(all_test_losses)
    print(f"Epoch {epoch} | batch train loss: {loss} | test loss: {test_loss}")
    make_plot_de(deep_ensemble,data_type,epoch)


################# Code Completed Executing #####################
#####################################################################
#####################################################################
#####################################################################
