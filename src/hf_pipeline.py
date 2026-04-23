from datasets import load_dataset, load_dataset_builder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_hf_loader(split, dataset, batch_size):
    dataset = load_dataset(dataset, split=split)

    if split=="train":
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
    else:
        # Validation should have standardized images
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_fn(examples):
        # Process image into tensors
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
        # To ensure images are not returned
        return {
            "pixel_values":examples["pixel_values"],
            "label":examples["label"]
        }   
    dataset.set_transform(transform_fn)

    is_train = (split == "train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train,num_workers=0, pin_memory=True)

def get_class_names():
    # Fetch the label names
    ds_builder = load_dataset_builder("ethz/food101")
    return ds_builder.info.features["label"].names