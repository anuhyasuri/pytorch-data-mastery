import torch
import time

# This automatically handles Mac (mps), Colab (cuda), or standard CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Code is running on: {device}")


#Stress test
if device == "cuda":
    # This prevents the 'NoneType' error by ensuring _C is populated.
    _ = torch.zeros(1, device=device) 
    torch.cuda.synchronize()

    x=torch.randn(10000,10000).to(device)
    y=torch.randn(10000,10000).to(device)
    start_time = time.perf_counter()

    z = torch.mm(x,y)
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    print(f"CUDA Handshake successful. Matrix Multiplication took {duration_ms:.2f}ms")
    print(f"Verification (Sample result): {z[0,0].item():.4f}")
else:
    print("CUDA not detected. Skipping GPU stress test.")


