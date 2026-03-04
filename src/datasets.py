import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir(string): Path to the directory where images are stored
        transform(callable, optional): Optional transform to be applied
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Adding label names from the folder names in images data
        # This below condition is to only include directories and ignore anything that starts with '.'
        # sorted() helps in maintaining the index for each class even if the folder order is different
        self.class_names = sorted([f for f in os.listdir(root_dir) 
                                  if os.path.isdir(os.path.join(root_dir, f)) and not f.startswith('.')
                                  ])
        self.class_idx = {cls: i for i, cls in enumerate(self.class_names)}
        print(self.class_idx)

MyImageDataset(root_dir = "data/archive/seg_train/seg_train", transform = None)

print("Done!")