import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from load_data import *

BATCH_SIZE = 64

class RoadDataset(Dataset):
    """
    Class to store the training road dataset and load them with the DataLoader
    """
    def __init__(self, data, labels):
        # Store the images and groundtruth
        self.images, self.gd_truth = data, labels

    def __len__(self): # Function to use the DataLoader
        return len(self.images)

    def __getitem__(self, idx): # Function to use the DataLoader to get one batch of images
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        #patched_image = np.transpose(self.images[idx], (2, 0, 1))
        #patched_image = torch.from_numpy(patched_image)
        patched_image = torch.from_numpy(self.images[idx]).float()

        #gt = torch.from_numpy(self.gd_truth[idx]*255).float().unsqueeze(0)
        gt = torch.from_numpy(np.asarray(self.gd_truth[idx]))

        return patched_image, gt


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding="same")
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same")
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=1600, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = torch.flatten(x,start_dim=1, end_dim= -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x



def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    loss_history = []
    accuracy_history = []
    lr_history = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 3, 1, 2)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = torch.round(output, decimals=0)

        correct = pred.eq(target.view_as(pred)).sum().item()
        loss_float = loss.item()
        accuracy_float = correct / len(data)

        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        lr_history.append(scheduler.get_last_lr()[0])
        '''
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )'''

    return loss_history, accuracy_history, lr_history


@torch.no_grad()
def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data = data.permute(0, 3, 1, 2)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)

        pred = torch.round(output, decimals=0)
        correct += pred.eq(target.view_as(pred)).item()

    test_loss /= len(val_loader.dataset)


    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct / len(val_loader.dataset)


def run_training(model_factory, num_epochs, optimizer_kwargs, train_loader, val_loader, device="cuda",):
    # ===== Data Loading =====
    #train_loader, val_loader = get_dataloaders(**data_kwargs)

    # ===== Model, Optimizer and Criterion =====
    model = model_factory
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    #criterion = torch.nn.functional.cross_entropy
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(len(train_loader.dataset) * num_epochs) // train_loader.batch_size,
    )

    # ===== Train Model =====
    lr_history = []
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, lrs = train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device)
        train_loss_history.extend(train_loss)
        train_acc_history.extend(train_acc)
        lr_history.extend(lrs)

        val_loss, val_acc = validate(model, device, val_loader, criterion)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

    # ===== Plot training curves =====
    n_train = len(train_acc_history)
    t_train = num_epochs * np.arange(n_train) / n_train
    t_val = np.arange(1, num_epochs + 1)

    plt.figure(figsize=(6.4 * 3, 4.8))
    plt.subplot(1, 3, 1)
    plt.plot(t_train, train_acc_history, label="Train")
    plt.plot(t_val, val_acc_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 2)
    plt.plot(t_train, train_loss_history, label="Train")
    plt.plot(t_val, val_loss_history, label="Val")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 3, 3)
    plt.plot(t_train, lr_history)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    '''
    # ===== Plot low/high loss predictions on validation set =====
    points = get_predictions(
        model,
        device,
        val_loader,
        partial(torch.nn.functional.cross_entropy, reduction="none"),
    )
    points.sort(key=lambda x: x[1])
    plt.figure(figsize=(15, 6))
    for k in range(5):
        plt.subplot(2, 5, k + 1)
        plt.imshow(points[k][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[k][3])} pred={int(points[k][2])}")
        plt.subplot(2, 5, 5 + k + 1)
        plt.imshow(points[-k - 1][0][0, 0], cmap="gray")
        plt.title(f"true={int(points[-k-1][3])} pred={int(points[-k-1][2])}")
    '''

    torch.save(model.state_dict(), 'models/last_model.pth')

    return sum(train_acc) / len(train_acc), val_acc




def main():
    training_data, training_labels = load_data()

    print('-----data loaded-----')
    # Split in training and validation set
    size_train = int(training_data.shape[0]*0.8)

    train_data = training_data[:size_train, :, :, :]
    train_labels = training_labels[:size_train, :]

    validation_data = training_data[size_train:, :, :, :]
    validation_labels = training_labels[size_train:, :]
    print('-----splitting done-----')

    # Build training and validation datasets and dataloaders
    train_set = RoadDataset(train_data, train_labels)
    validation_set = RoadDataset(validation_data, validation_labels)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_set, 1, shuffle=False)
    print('-----data loading done-----')

    #model_factory = lambda: get_cnn(image_size=16)
    model_factory = SimpleNet()

    # Define relevant variables for the ML task
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_kwargs = dict(lr=5e-4, weight_decay=1e-3,)

    run_training(model_factory=model_factory, num_epochs=num_epochs, optimizer_kwargs=optimizer_kwargs, train_loader=train_loader, val_loader=validation_loader , device=device)

    print('-----training done-----')


#main()







