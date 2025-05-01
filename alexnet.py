'''
IMPLEMENTATION OF ALEXNET
'''
from tkinter import Image

#Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import os
from PIL import Image
#Configuring the Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Initialize of Training and Validation Data Parameters

def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_data_size= 0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    #Define transformation of validation data
    validation_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])
    #Dataset for Training
    train_dataset = datasets.ImageFolder(
        root = os.path.join(data_dir, 'images'), transform = train_transform

    )
    #Dataset for Validation
    validation_dataset = datasets.ImageFolder(
        root = os.path.join(data_dir, 'images'), transform = validation_transform

    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = max(1, int(np.floor(valid_data_size * num_train)))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler )

    valid_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    return (train_loader, valid_loader)

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    image_folder_path = os.path.join(data_dir, 'images')

    dataset = datasets.ImageFolder(
        root=image_folder_path,
        transform=transform
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

# Update train and validation loader call
train_loader, valid_loader = get_train_valid_loader(
    data_dir='./data_dir',  # This points to the main directory that contains 'images'
    batch_size=10, #used to be 64
    augment=False,           # Set to True if you want data augmentation
    random_seed=1
)

# Update test loader call
test_loader = get_test_loader(
    data_dir='./data_dir',  # This points to the main directory that contains 'images'
    batch_size=10
)


#Class for the AlexNet Neural Network
class AlexNet(nn.Module):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()
        #Number of Classes --> Number of Different Images to Train OUr Model On

        #Basic 5 Convolution Layers, 3 fully connected layers of AlexNet
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    #Forward Propagation
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":

    # Definition of Hyperparameter
    num_classes = 9  # Number of different classes to train on
    num_epochs = 10  # Number of times trained
    batch_size = 10
    learning_rate = 0.001

    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Type of Loss Function
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    # Epoch -> One iteration through the entire training dataset
    total_step = len(train_loader)  # This is number of batches (steps) per epoch

    # Actually training our model

    # Go over the number of epochs (how many times model sees the whole dataset)
    for epoch in range(num_epochs):
        # Iterate through batches in train_loader
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Go backward (propagation) and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            # Validation
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    del images, labels, outputs

                print('Accuracy of the network on the {} validation images: {} %'.format(81, 100 * correct / total))

            print("TRAINING DONE. PERFORMING TESTING......")  # Now moving on to testing

            # 81 -> Total number of images