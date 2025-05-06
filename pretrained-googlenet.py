"""
Fine-tune pretrained GoogLeNet on small dataset
"""
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import googlenet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(11)

# Data preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

dataset = ImageFolder(root=os.path.join('./data_dir', 'images'), transform=transform)

if __name__ == "__main__":
    print(f"Dataset size: {len(dataset)}")
    print("Classes:", dataset.class_to_idx)
    num_classes = len(dataset.classes)

    # Split sizes
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Visualize a batch
    for images, _ in train_loader:
        plt.figure(figsize=(12, 6))
        plt.axis("off")
        plt.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        plt.show()
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained GoogLeNet
    model = googlenet(pretrained=True, aux_logits=True)

    # Freeze all layers except classifier heads
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classifiers
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    # Unfreeze classifier layers
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.aux1.parameters():
        param.requires_grad = True
    for param in model.aux2.parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=15):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs-1}")
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            outputs, aux1, aux2 = model(inputs)
                            loss = criterion(outputs, labels)
                            loss += 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

        model.load_state_dict(best_model_wts)
        print(f"Best val Acc: {best_acc:.4f}")
        return model

    model = train_model(model, {"train": train_loader, "val": val_loader}, criterion, optimizer, num_epochs=10)