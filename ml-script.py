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
        modules = []
        for layer in layers:
            modules.append(nn.Linear(layer[0], layer[1]))
            modules.append(nn.LeakyReLU())
        modules.pop()
        self.runModel = nn.Sequential(*modules).to(device)
    def forward(self, x):
        pred = self.runModel(x)
        return pred
    def getDevice(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import sys
from shutil import copyfile
import os
import itertools
from livelossplot import PlotLosses
liveloss = PlotLosses()

# ----------CONFIG---------
learning_rate = 1e-5
epochs_train = 3000
loss_fn = torch.nn.MSELoss()
#---------END CONFIG----------
# print(list(filter(lambda z: True if z[0] < 0.1 else False, ap_input)))
model = PredictionModel([[1,200],[200,100],[100,1]])
resuming = False
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
date_arr = []
for i in range(1):
    date_arr.append(list(range(128)))
x = Variable(torch.from_numpy(np.array(list(map(lambda inp: [inp], np.array(date_arr).flatten()))))).to(torch.float).to(model.getDevice()) # months after the first date, repeated per subject
# y = Variable(torch.from_numpy(np.array(list(map(lambda inp: [inp], np.array(torch_ap_input[4]).flatten()))))).to(torch.float).to(model.getDevice()) # ap factor
y = Variable(torch.from_numpy(np.array(list(map(lambda inp: [inp], np.array(list(map(lambda x: np.sin(x), x))).flatten()))))).to(torch.float).to(model.getDevice()) # ap factor
plt.plot(x,y)
plt.show()
BATCH_SIZE = 64

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=0)
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
logs = []
model.train()
for t in range(epochs_train):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        logs={}
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = model(b_x)     # input x and predict based on x

        loss = loss_fn(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        logs['train log loss'] = loss.item()
    liveloss.update(logs)
plt.clf()
print('evaling now')
model.eval()
x1 = []
y1 = []
for val in range(195):
    x1.append(val)
    inp = Variable(torch.from_numpy(np.array([val]))).to(torch.float).to(model.getDevice())
    output=model(inp)
    y1.append(output.item())

plt.plot(x1, y1)
plt.draw()
plt.show()
# %%
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import imageio

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(x) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), y.data.numpy(), color = "blue")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.savefig('curve_2.png')
plt.show()

# another way to define a network
net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

my_images = []
fig, ax = plt.subplots(figsize=(16,10))

# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if step == 1:
            # plot and show learning process
            plt.cla()
            ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
            ax.set_xlabel('Independent variable', fontsize=24)
            ax.set_ylabel('Dependent variable', fontsize=24)
            ax.set_xlim(-11.0, 13.0)
            ax.set_ylim(-1.1, 1.2)
            ax.scatter(b_x.data.numpy(), b_y.data.numpy(), color = "blue", alpha=0.2)
            ax.scatter(b_x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
            ax.text(8.8, -0.8, 'Epoch = %d' % epoch,
                    fontdict={'size': 24, 'color':  'red'})
            ax.text(8.8, -0.95, 'Loss = %.4f' % loss.data.numpy(),
                    fontdict={'size': 24, 'color':  'red'})

            # Used to return the plot as an image array 
            # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            my_images.append(image)

    


# save images as a gif    
imageio.mimsave('./curve_2_model_3_batch.gif', my_images, fps=12)


fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
ax.set_xlabel('Independent variable', fontsize=24)
ax.set_ylabel('Dependent variable', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)
prediction = net(x)     # input x and predict based on x
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
plt.savefig('curve_2_model_3_batches.png')
plt.show()
