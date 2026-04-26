import torch
import time

device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"targeted device {device}")

#Diagnostics
print(f"1. Is MPS built into this PyTorch? {torch.backends.mps.is_built()}")
print(f"2. Is MPS available on this machine? {torch.backends.mps.is_available()}")

import platform
print(f"3. macOS Version: {platform.mac_ver()[0]}")

#Stress test
if device.type == "mps":
    # Perform a tiny operation FIRST to initialize the backend
    # This prevents the 'NoneType' error by ensuring _C is populated.
    _ = torch.zeros(1, device=device) 
    torch.mps.synchronize()

    x=torch.randn(2000,2000).to(device)
    y=torch.randn(2000,2000).to(device)
    start_time = time.perf_counter()

    z = torch.mm(x,y)
    torch.mps.synchronize()

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    print(f"GPU Handshake successful. Matrix Multiplication took {duration_ms:.2f}ms")
    print(f"Verification (Sample result): {z[0,0].item():.4f}")
else:
    print("MPS not detected. Skipping GPU stress test.")
#----------------------------------------------------
# Run this code to check the direct initiation to MPS
# import torch
# import time
# print("Initializing MPS on Mac Tahoe")
# start = time.time()
# try:
#     # Manually try to create a tensor on mps
#     test_tensor = torch.ones(1).to("mps")
#     print("SUCCESS: MPS is active via manual override!")
#     print(f"Warmup took: {time.time()-start:.2f}s")
# except Exception as e:
#     print(f"FAIL: {e}")

