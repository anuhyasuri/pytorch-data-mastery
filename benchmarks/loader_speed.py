import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def run_benchmark(num_workers, pin_memory, total_batches = 50):
  transform = transforms.Compose(
    [
      transforms.Resize((224,224)),
      transforms.ToTensor()
    ]
  )
  
  dataset = datasets.FakeData(
    size = 5000, image_size=(3,224,224), num_classes=101, transform=transform
  )

  loader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
  )

  iterator = iter(loader)
  _=next(iterator)

  start_time = time.perf_counter()

  for i in range(total_batches):
    try:
      images, labels = next(iterator)
      images = images.to("cuda")
    except StopIteration:
      break

  end_time = time.perf_counter()
  duration = end_time - start_time
  batches_per_second = total_batches/duration

  print(f"Configuration: num_workers={num_workers} | pin_memory={pin_memory}")
  print(f"Time for {total_batches} batches: {duration:.4f} seconds")
  print(f"Throughput: {batches_per_second:.2f} batches/sec\n")
  return batches_per_second

if __name__ == "__main__":
  print("Starting Data Loader Speed Benchmarks on CUDA\n")

  # Test 1: Single-threaded (Baseline)
  baseline = run_benchmark(num_workers=0, pin_memory=False)

  # Test 2: Multi-threaded + Memory Pinning Optimized
  optimized = run_benchmark(num_workers=2, pin_memory=True)

  improvement = (optimized - baseline) / baseline * 100
  print(f"Overall Pipeline Throughput Improvement: {improvement:.2f}%")

