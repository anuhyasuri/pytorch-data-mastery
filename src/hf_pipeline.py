from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def get_hf_loader(dataset, batch_size):
    dataset = load_dataset(dataset, split="train[:10%]")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    def transform_fn(examples):
        # Process image into tensors
        examples["pixel_values"] = [data_transforms(img.convert("RGB")) for img in examples["image"]]
        # To ensure images are not returned
        return {
            "pixel_values":examples["pixel_values"],
            "label":examples["label"]
        }
    dataset.set_transform(transform_fn)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=0)