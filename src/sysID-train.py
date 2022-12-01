import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from models import sysID

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Dataset
batch_size = 4096

data_path = "./CPSC532J_FinalProject/data/trajectories.npy"
label_path = "./CPSC532J_FinalProject/data/labels.npy"

dataset = TensorDataset(torch.Tensor(np.load(data_path)),
                                     torch.Tensor(np.load(label_path)))
dataset_full_len = len(dataset)
train_size = int(0.8 * dataset_full_len)
val_size = int(0.1 * dataset_full_len)
test_size = dataset_full_len - train_size - val_size

trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True) 

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

Identifier = sysID(step_limit=50)
Identifier = Identifier.to(device)
optimizer = torch.optim.Adam(Identifier.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = torch.nn.MSELoss()

epochs = 125
epoch_printfreq = 1 # Print output frequency

train_losses = []
val_losses = []
f1scores = []

# Training
for epoch in range(epochs):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_f1scores = []    
    # if epoch % epoch_printfreq == 0:
    #     print("Starting epoch: ", epoch, "...")
    # Training
    Identifier.train()
    print("Epoch: ", epoch)
    for batch_idx, batch in enumerate(tqdm(trainloader)):
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        pred = Identifier(x)
        loss = criterion(pred, y)
        epoch_train_losses.append(loss.detach().cpu())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if batch_idx >= 1000:
        #     break

    # Validation
    Identifier.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            x = batch[0]
            y = batch[1]
            
            x = x.to(device)
            y = y.to(device)

            pred = Identifier(x)
            loss = criterion(pred, y)

            epoch_val_losses.append(loss.detach().cpu())
            # if batch_idx >= 1000:
            #     break

    epoch_mean_train_loss = np.mean(epoch_train_losses)
    train_losses.append(epoch_mean_train_loss)
    epoch_mean_val_loss = np.mean(epoch_val_losses)
    val_losses.append(epoch_mean_val_loss)

    if epoch % epoch_printfreq == 0:
        print("Epoch", epoch, 
              "mean train loss: ", epoch_mean_train_loss, 
              ", mean val loss: ", epoch_mean_val_loss)

# Save the model
torch.save(Identifier, "./CPSC532J_FinalProject/src/model_checkpoints/sysID.pth")

# Plot
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.suptitle("Identifier Loss")
plt.legend()
plt.savefig("./CPSC532J_FinalProject/src/logs/sysID/training_curves.jpg")
plt.show()  

