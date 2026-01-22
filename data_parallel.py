import time

import torch
import torch.nn as nn


# Assuming FashionCNN is defined here or imported
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(14 * 14 * 16, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Instantiate and Wrap Model
model = FashionCNN()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# 3. Dummy Batch (Adjust size for testing)
batch_size = 64
n_reps = 100
batch = torch.randn(batch_size, 1, 28, 28).to(device)

# 4. Timing Inference
model.eval()
with torch.no_grad():
    # Warm up
    for _ in range(10):
        _ = model(batch)

    start = time.time()
    for _ in range(n_reps):
        out = model(batch)
    # Ensure GPUs finish before stopping clock
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

print(f"Total time for {n_reps} reps: {end - start:.4f}s")
