#Torch
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split
from torch import nn
from torch.utils.data import DataLoader
import os
# Distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#Project
from Model import CustomModel
from Dataset import ObjectPositionDataset
from Utils import *
from Train import *

# Constants
ddp_setup()
BATCH_SIZE = 12
LEARNING_RATE = 0.001
EPOCHS = 50
DATA_ROOT_DIR = 'Dataset'  # Replace with your image directory path

# Step 1: Set Up Dataset and Dataloader
transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = ObjectPositionDataset(csv_file='combined_dataset.csv', root_dir=DATA_ROOT_DIR, transform=transform)
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
local_rank = int(os.environ["LOCAL_RANK"])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,sampler=DistributedSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 2: Create Model Instance
model = CustomModel()
model = DDP(model, device_ids=[local_rank])

# Step 3: Define Loss Function and Optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = []
val_losses = []


# Step 4 : Train and Validate the Model
train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, EPOCHS,checkpoint_path="CombinedImage/checkpoint", save_epoch_interval=1)
# Step 5: Save Checkpoints
if os.environ["GLOBAL_RANK"] == 0:
    checkpoint = {
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()  # Assuming optimizer is defined
    }
    save_checkpoint(checkpoint, filename="CombinedImage/checkpoint/final_checkpoint.pth.tar")
# Step 6: Cleanup
destroy_process_group()