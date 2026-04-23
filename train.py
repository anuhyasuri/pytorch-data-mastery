import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.datasets import MyImageDataset
from src.utils import visualize_data
from src.hf_pipeline import get_hf_loader, get_class_names

DATASET_MODE = "kaggle" # or "huggingface"
def main():
    if DATASET_MODE == "huggingface":
        # Hugging face dataset
        is_hf = True
        hf_loader = get_hf_loader(split = "train", dataset = "ethz/food101", batch_size = 32)
        class_names = get_class_names()
        batch = next(iter(hf_loader))
        print(f"Batch shape: {batch['pixel_values'].shape}")
        print("Visualizing Huggingface data")

    else:
        is_hf = False
        # Kaggle dataset
        folder_type = "seg_test" # or "seg_test"
        if folder_type == "seg_train":
            # This prevents overfitting by ensuring the model never sees the exact same pixel grid twice.
            transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            # Normalizing to ImageNet distribution
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif folder_type == "seg_test":
            # Validation should have standardized images
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])         
            ])
        root_dir = f"data/archive/{folder_type}/{folder_type}"
        train_dataset = MyImageDataset(root_dir = root_dir, transform = transform)
        batch = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)
        class_names = train_dataset.class_names
        images, labels = next(iter(batch))
        print(f"Batch shape: {images.shape}")
        print("Visualizing Pytorch data")

    visualize_data(batch, class_names, is_hf)
    print("Done!")


if __name__ == '__main__':
    main()