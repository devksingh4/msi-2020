import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
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

model = torch.load('models/final_model.tar',map_location="cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
# # inp = Variable(torch.from_numpy(np.array([float(input("enter: "))]))).to(torch.float).to(model.getDevice())
# output=model(inp)
# print(output.item())
x = []
y = []
for val in range(0, 101):
    val = val / 100
    x.append(val)
    inp = Variable(torch.from_numpy(np.array([val]))).to(torch.float).to(model.getDevice())
    output=model(inp)
    y.append(output.item())

plt.plot(x, y)
plt.draw()
plt.show()

