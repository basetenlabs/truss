"""
MNIST Training Example

A simple CNN classifier for the MNIST handwritten digits dataset.
This example demonstrates:
- Using BT_RW_CACHE_DIR for caching downloaded data
- Using BT_CHECKPOINT_DIR for saving model checkpoints
- Basic PyTorch training loop
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data directory - use cache so data persists between runs
data_dir = os.path.join(os.environ["BT_RW_CACHE_DIR"], "mnist")

# Data loaders
transform = transforms.ToTensor()
train_loader = DataLoader(
    torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    ),
    batch_size=512,
    shuffle=True,
)
test_loader = DataLoader(
    torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    ),
    batch_size=512,
    shuffle=False,
)


# Simple CNN model
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MNISTClassifier().to(device)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# Checkpoint directory - Baseten will save these for you
checkpoint_dir = os.environ["BT_CHECKPOINT_DIR"]
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
print("Starting MNIST training...")
for epoch in range(5):
    # Train
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Epoch: {epoch + 1}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    correct = sum(
        model(data.to(device)).argmax(1).eq(target.to(device)).sum().item()
        for data, target in test_loader
    )
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Epoch {epoch + 1}: Accuracy: {accuracy:.2f}%")

    # Save checkpoint
    torch.save(
        model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth")
    )

print("Training completed!")
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final.pth"))
