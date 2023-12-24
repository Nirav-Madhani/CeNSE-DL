import pandas as pd
from glob import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ObjectPositionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, stack_mode='vertical'):
        self.annotations = pd.read_csv(csv_file,index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.stack_mode = stack_mode  # 'vertical' or 'horizontal'

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, f"{self.annotations.iloc[index, 0]}_cam")
        image1 = Image.open(img_name + "1.jpg")
        image2 = Image.open(img_name + "2.jpg")

        # Stack images
        if self.stack_mode == 'vertical':
            combined_image = Image.new('RGB', (image1.width, image1.height + image2.height))
            combined_image.paste(image1, (0, 0))
            combined_image.paste(image2, (0, image1.height))
        else:  # horizontal
            combined_image = Image.new('RGB', (image1.width + image2.width, image1.height))
            combined_image.paste(image1, (0, 0))
            combined_image.paste(image2, (image1.width, 0))

        y_label = torch.tensor(self.annotations.iloc[index, 1:4].values.astype(float)).float()

        if self.transform:
            combined_image = self.transform(combined_image)

        return combined_image, y_label