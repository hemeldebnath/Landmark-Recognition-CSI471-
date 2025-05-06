import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Choose device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization functions
def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_data_size=0.1, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.2023, 0.1994, 0.2010])

    # Apply augmentation if enabled
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]) if augment else transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    base_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'images'), transform=None)

    num_samples = len(base_dataset)
    indices = list(range(num_samples))
    split = int(np.floor(valid_data_size * num_samples))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(base_dataset, valid_indices)
    train_dataset.dataset.transform = train_transform
    valid_dataset.dataset.transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor(), normalize])
    image_folder_path = os.path.join(data_dir, 'images', 'test_data_class')
    dataset = datasets.ImageFolder(root=image_folder_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Define the neural network modules
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.squeeze(x))
        return self.relu(torch.cat([self.expand1x1(x), self.expand3x3(x)], 1))


class MiniInception(nn.Module):
    def __init__(self, in_channels):
        super(MiniInception, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 32, 1)
        self.conv3x3 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, 32, 5, padding=2)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        return torch.cat([out1, out2, out3], 1)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


class NHTNet(nn.Module):
    def __init__(self, num_classes=10):
        super(NHTNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block1 = MiniInception(64)
        self.block2 = FireModule(96, 16, 64)
        self.block3 = DepthwiseSeparableConv(128, 128)
        self.block4 = FireModule(128, 16, 64)
        self.block5 = MiniInception(128)
        self.final_conv = nn.Conv2d(96, num_classes, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.final_conv(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)


# Model initialization
train_loader, valid_loader = get_train_valid_loader(data_dir='./data_dir', batch_size=10, augment=True, random_seed=1)
test_loader = get_test_loader(data_dir='./training_data', batch_size=10)

model = NHTNet(num_classes=10).to(device)

# Hyperparameters
num_epochs = 30
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)


#Variables for accuracy, loss

train_losses = []
train_accuracies = []
val_accuracies = []


criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()

        # Track training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Evaluate validation accuracy
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    # Save metrics
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Loss: {avg_loss:.4f}, "
          f"Train Acc: {train_accuracy:.2f}%, "
          f"Val Acc: {val_accuracy:.2f}%")


# Plot Loss per Epoch
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Plot Accuracy per Epoch
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
