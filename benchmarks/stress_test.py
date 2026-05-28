import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def run_stress_test():
  print("Initializing stress test")
  transform = transforms.Compose(
    [
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ]
  )

  dataset = datasets.FakeData(size = 2500, image_size = (3,224,224), num_classes = 101, transform=transform)

  loader = DataLoader(dataset, batch_size = 64, shuffle=False, num_workers=0, pin_memory=True)

  total_brightness = 0.0
  total_pixels = 0
  batch_count = 0

  start_time = time.perf_counter()

  print(f"Iterating through {len(loader)} batches...")

  for images, _ in loader:
    images = images.to("cuda")
    total_brightness += torch.sum(images).item()
    total_pixels += images.numel()

    batch_count +=1
    if batch_count %10 ==0:
      print(f"Processed batch {batch_count}/{len(loader)}... Pipeline stable.")

  end_time = time.perf_counter()

  global_average_brightness = total_brightness / total_pixels
  duration = end_time - start_time
    
  print("\n --- STRESS TEST COMPLETE ---")
  print(f"Total processing time: {duration:.2f} seconds")
  print(f"Engineered Feature - Global Average Brightness: {global_average_brightness:.4f}")

if __name__ == "__main__":
    run_stress_test()
