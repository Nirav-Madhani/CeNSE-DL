import torch
from torch.distributed import init_process_group
import os
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    global device
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))