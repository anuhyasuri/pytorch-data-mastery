import os
import torch
from torch.utils.data import Dataset
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
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        # Adding label names from the folder names in images data
        # This below condition is to only include directories and ignore anything that starts with '.'
        # sorted() helps in maintaining the index for each class even if the folder order is different
        self.class_names = sorted([f for f in os.listdir(root_dir) 
                                  if os.path.isdir(os.path.join(root_dir, f)) and not f.startswith('.')
                                  ])
        self.class_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        # Walk through folders to get the image paths
        for cls in self.class_names:
            cls_path = os.path.join(root_dir, cls)
            per_class_count = 0
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(valid_extensions): # This check will ensure to read only images
                    self.image_paths.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_idx[cls])
                    per_class_count += 1
            print(f"Class {cls}: loaded {per_class_count} images")
        print("Dataset Loaded")
        print(f"Path: {root_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Total Images: {len(self.image_paths)}")
        print("-" * 20 + "\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)