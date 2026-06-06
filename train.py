import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.datasets import MyImageDataset
from src.utils import visualize_data
from src.hf_pipeline import get_hf_loader, get_class_names
from src.model import FoodClassifierBaseline
from src.engine import train_one_epoch, validate_one_epoch

DATASET_MODE = "huggingface" # or "kaggle"

def main():
  print(f"Initializing training pipeline using {DATASET_MODE} data")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Executing on {device}")
  
  if DATASET_MODE == "huggingface":
    # Hugging face dataset
    is_hf = True
    train_loader = get_hf_loader(split = "train", dataset = "ethz/food101", batch_size = 32)
    val_loader = get_hf_loader(split = "validation", dataset = "ethz/food101", batch_size = 32)
    class_names = get_class_names()
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['pixel_values'].shape}")
    print("Visualizing Huggingface data")

  else:
    is_hf = False
    # Kaggle dataset
    # This prevents overfitting by ensuring the model never sees the exact same pixel grid twice.
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # Normalizing to ImageNet distribution
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation should have standardized images
    val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
    ])

    train_dataset = MyImageDataset(root_dir = "data/archive/seg_train/seg_train", transform = train_transform)
    val_dataset = MyImageDataset(root_dir = "data/archive/seg_test/seg_test", transform = val_transform)
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False, num_workers = 0)

    class_names = train_dataset.class_names
    images, labels = next(iter(batch))
    print(f"Batch shape: {images.shape}")
    print("Visualizing Pytorch data")

  visualize_data(batch, class_names, is_hf)
  print("Done!")


if __name__ == '__main__':
    main()