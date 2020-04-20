# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Notebook to import data from Google Trends given CSV files and scale them and perform ANOVA (and eventually Tukey HSD) analysis. 
# %% [markdown]
# ## Part 1: Statistical Analysis

# %%
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from matplotlib import pyplot as plt

# %% [markdown]
# Here, I'm going to use pandas to import data from all CSVs that i tell it to import data from into a pandas dataframe

# %%
files = ["ComputerScience", "Chemistry", "Biology", "Physics", "Macro", "MusicTheory", "Statistics", "Psych"]
data = []
for name in files: 
    i = files.index(name)
    full_rel_path = "raw-datasets/" + name + ".csv"
    data.append(pd.read_csv(full_rel_path, sep=",", header=None, skiprows=3)) # First 3 rows have headers that I don't want
    data[i] = data[i].drop(data[i].columns[0], axis=1)
    data[i] = data[i].replace('<1', '1')
    data[i].columns = ['AP ' + name + ' interest', name + ' interest']
    data[i]['AP ' + name + ' interest'] = pd.to_numeric(data[i]['AP ' + name + ' interest'],errors='coerce')
    data[i][name + ' interest'] = pd.to_numeric(data[i][name + ' interest'],errors='coerce')
    data[i] = data[i] / 100

# %% [markdown]
# The above exporting of the scaled dataset is occuring just to backup data in case of failure

# %%
dev_stats = []
for df in data: 
    ap_diff = []
    field_diff = []
    ap_mean = df[df.columns[0]].mean()
    field_mean = df[df.columns[1]].mean()
    for index, row in df.iterrows(): 
        ap_interest = row[df.columns[0]]
        field_interest = row[df.columns[1]]
        ap_diff.append((ap_mean - ap_interest) ** 2) # must be squared because the negative and positive versions cause issues. 
        field_diff.append((field_mean - field_interest)**2)
    dev_stats.append(pd.DataFrame(list(zip(ap_diff, field_diff)),columns =['AP Interest DEV Stat', 'Field DEV Stat']))
# %% [markdown]
# The above cell takes all of the data processed, calculated the DEV stat, and adds it to the dataframe

# %%
ap_input = []
field_input = []
for df in dev_stats:
    ap_input.append(np.array(df[df.columns[0]]))
    field_input.append(np.array(df[df.columns[1]]))
torch_ap_input = []
torch_field_input = []
for df in data:
    torch_ap_input.append(np.array(df[df.columns[0]]))
    torch_field_input.append(np.array(df[df.columns[1]]))
# print(scipy.stats.f_oneway(ap_input[0], ap_input[1], ap_input[2], ap_input[3], ap_input[4], ap_input[5], ap_input[6], ap_input[7], ap_input[8]))
# print(scipy.stats.f_oneway(field_input[0], field_input[1], field_input[2], field_input[3], field_input[4], field_input[5], field_input[6], field_input[7], field_input[8])


# %%
import torch
import torch.nn as nn
class PredictionModel(nn.Module):
    """Dynamically created PredictionModel based off of inputted parameters for layers"""
    def __init__(self, layers):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(PredictionModel, self).__init__()
        super(PredictionModel, self).__init__()
        modules = []
        for layer in layers:
            modules.append(nn.Linear(layer[0], layer[1]))
            modules.append(nn.ReLU(inplace=True))
        modules.pop()
        self.runModel = nn.Sequential(*modules).to(device)
    def forward(self, x):
        pred = self.runModel(x)
        return pred
    def getDevice(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# %%
import torch
import sys 
from torch.autograd import Variable
from shutil import copyfile
import os
import itertools
# ----------CONFIG---------
learning_rate = 1e-4
epochs_train = 100
loss_fn = torch.nn.L1Loss()
#---------END CONFIG----------
# print(list(filter(lambda z: True if z[0] < 0.1 else False, ap_input)))
model = PredictionModel([[1,1024],[1024,1024],[1024,1]])
resuming = False
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
if os.path.isfile('models/interim_model.tar'):
    try:
        checkpoint = torch.load('models/interim_model.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_cp = checkpoint['loss']
        resuming = True
    except RuntimeError:
        print("Could not import model. File exists")
        pr = input("Continue? [y/n]: ")
        if pr.lower() != "y":
            sys.exit(127)
x = Variable(torch.from_numpy(np.array(list(map(lambda inp: [inp], np.array(torch_ap_input).flatten()))))).to(torch.float).to(model.getDevice())
y = Variable(torch.from_numpy(np.array(list(map(lambda inp: [inp], np.array(torch_field_input).flatten()))))).to(torch.float).to(model.getDevice())
logs = []
# model.eval()
# print(model(Variable(torch.from_numpy(np.array(list(itertools.repeat(0.3, 1950))))).to(torch.float).to(model.getDevice())).data.cpu().numpy())
for t in range(epochs_train):
    if resuming:
        it = epoch + t
        loss = loss_cp
    else: 
        it = t
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    logs.append(float(loss.item()))
    if it % 100 == 99:
        torch.save({
            'epoch': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "models/interim_model.tar")
        print("Epoch: ", str(it), "Loss: ", logs[len(logs)-1])
        plt.plot(list(range(t+1)), logs)
        plt.draw()
        plt.pause(0.01)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
torch.save({
    'epoch': it,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, "models/interim_model.tar")
torch.save(model, 'models/final_model.tar')
plt.clf()


# %%
