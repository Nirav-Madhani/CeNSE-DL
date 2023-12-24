#Torch
from tqdm import tqdm
import os
#Project
from Utils import *

# Constants

def train_and_validate(model, train_loader, val_loader, loss_fn, optimizer, epochs, epoch_offset=0,checkpoint_path="model_checkpoint", save_epoch_interval=5):
    global local_rank, train_losses, val_losses
    # Load latest checkpoint from given directory if exists
    if os.path.isfile(checkpoint_path+"/epoch2.pth.tar"):
        if input("Checkpoint found. Load from checkpoint? (y/N) ") == 'y':
          checkpoint = torch.load(checkpoint_path+"/epoch2.pth.tar")
          model.load_state_dict(checkpoint['state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print(f'Model loaded from {checkpoint_path}.')
    elif input("No checkpoint found. Train from scratch? (y/N) ") == 'y':
        pass
    else:
        raise FileNotFoundError

    for epoch in range(epoch_offset, epochs+epoch_offset):
        model.train()

        loop = tqdm(train_loader, leave=True)
        for (image, labels) in loop:
            image = image.to(local_rank)
            labels = labels.to(local_rank)

            preds = model(image)
            loss = loss_fn(preds, labels.float())  # Ensure labels are float

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss=loss.item())
            train_losses.append(loss.item())
            #save train losses to file
            with open(f'train_losses_{epoch_offset}.txt', 'w') as f:
                for item in train_losses:
                    f.write("%s\n" % item)

        if (epoch + 1) % save_epoch_interval == 0 and os.environ["GLOBAL_RANK"] == 0:
            checkpoint = {
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict()  # Assuming optimizer is defined
            }
            save_checkpoint(checkpoint, filename=f"{checkpoint_path}/epoch{epoch+1}.pth.tar")

        model.eval()
        with torch.no_grad():
            # Validation loop
            val_loss = 0
            for (image, labels) in val_loader:
                image = image.to(local_rank)
                labels = labels.to(local_rank)
                preds = model(image)
                val_loss += loss_fn(preds, labels.float()).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            #save val losses to file
            with open(f'val_losses_{local_rank}.txt', 'w') as f:
                for item in val_losses:
                    f.write("%s\n" % item)
            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

