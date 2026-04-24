# import torch

# device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"targeted device {device}")

# #Diagnostics
# print(f"1. Is MPS built into this PyTorch? {torch.backends.mps.is_built()}")
# print(f"2. Is MPS available on this machine? {torch.backends.mps.is_available()}")

# import platform
# print(f"3. macOS Version: {platform.mac_ver()[0]}")

# #Stress test
# if device == "mps":
#     x=torch.randn(2000,2000).to(device)
#     y=torch.randn(2000,2000).to(device)
#     start = torch.mps.Event(enable_timing = True)
#     end = torch.mps.Event(enable_timing = True)

#     start.record()
#     z = torch.mm(x,y)
#     torch.mps.synchronize()

#     print(f"GPU Handshake successful. Matrix Multiplication took {start.elapsed_time(end):.2f}ms")

import torch
try:
    # Manually try to create a tensor on mps
    test_tensor = torch.ones(1).to("mps")
    print("🚀 SUCCESS: MPS is active via manual override!")
except Exception as e:
    print(f"❌ FAIL: {e}")

