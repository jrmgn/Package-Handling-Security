import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models import SimpleCNN
import os

def train_cnn():
    model = SimpleCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting CNN Baseline training...")

    dummy_inputs = torch.randn(10, 3, 128, 128)
    dummy_labels = torch.randint(0, 2, (10, 1)).float()
    
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/3 | Loss: {loss.item():.4f}")

    os.makedirs('../models', exist_ok=True)
    torch.save(model.state_dict(), "../models/cnn_baseline.pth")
    print("Success: Saved models/cnn_baseline.pth")

if __name__ == "__main__":
    train_cnn()
