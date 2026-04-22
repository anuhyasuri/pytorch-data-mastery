import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.datasets import MyImageDataset
from src.utils import visualize_data
from src.hf_pipeline import get_hf_loader
def main():

    # Pytorch dataset
    my_transforms = transforms.Compose([
    transforms.Resize((224,224)), # Standard resizing
    transforms.ToTensor(), # Convert to a tensor
    transforms.Normalize( # Imagenet Normalization
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

    train_dataset = MyImageDataset(root_dir = "data/archive/seg_train/seg_train", transform = my_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)

    images, labels = next(iter(train_dataloader))
    print(f"Batch shape: {images.shape}")
    print("Visualizing data")
    visualize_data(train_dataloader, train_dataset.class_names)
    print("Done!")

    # Hugging face dataset
    hf_loader = get_hf_loader(dataset = "ethz/food101", batch_size = 32)
    data_batch = next(iter(hf_loader))
    print("HF",data_batch)

if __name__ == '__main__':
    main()