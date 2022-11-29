import torch
from torch import nn
import torch.nn.functional as F

# Initial survivor network
class Survivor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# System ID network
class sysID(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 50 * 10
        self.l1 = nn.Linear(self.input_size, 1026)
        self.l2 = nn.Linear(1026, 512)
        self.l3 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

# Gen alg. generalist Network
class GA_Generalist(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4, bias=False)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=0)
        return x