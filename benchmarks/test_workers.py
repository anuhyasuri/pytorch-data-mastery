import sys
import os
# This adds the parent directory (root) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # To help the script access /src
import time
from torch.utils.data import DataLoader
from src.datasets import MyImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt

def benchmark_workers(worker_count):
    my_transforms = transforms.Compose([
    transforms.Resize((224,224)), # Standard resizing
    transforms.ToTensor(), # Convert to a tensor
    transforms.Normalize( # Imagenet Normalization
        mean = [0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])
    dataset = MyImageDataset(root_dir = "data/archive/seg_train/seg_train", transform = my_transforms)
    results = []
    for count in worker_count:
        dataloader = DataLoader(dataset, batch_size = 32, shuffle = True, num_workers = count)

        start = time.time()
        for _ in dataloader: pass 
        end = time.time()
        duration = end - start
        # print(f"Num of workers: {count:<10} | Time: {duration:.2f}")
        results.append(duration)

    plt.figure(figsize = (8,5))
    plt.bar(worker_count, results)
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.title('DataLoader Performance: Workers vs Time')
    plt.savefig('benchmark_results.png')

if __name__ == '__main__':
    benchmark_workers([0,2,4,6,8])
