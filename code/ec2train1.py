import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.set_num_threads(2)

def test(model, test_loader, criterion):
    model.eval()
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.double() / len(test_loader.dataset)
    print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {total_acc:.4f}")

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    epochs = 15
    best_loss = float('inf')
    loss_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = validation_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(data_loader, desc=f"{phase} progress"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(f"Batch Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                loss_counter = 0
            elif phase == 'valid':
                loss_counter += 1

        if loss_counter == 1:
            print("Early stopping due to no improvement in validation loss.")
            break

    return model

def net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 133)
    )
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_data_loader, test_data_loader, validation_data_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    learning_rate = 1e-4

    train_loader, test_loader, validation_loader = create_data_loaders('dogImages', batch_size)
    model = net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    print("Starting Model Training")
    model = train(model, train_loader, validation_loader, criterion, optimizer, device)

    print("Saving trained model...")
    os.makedirs('TrainedModels', exist_ok=True)
    torch.save(model.state_dict(), 'TrainedModels/model.pth')
    print("Model saved successfully.")

    print("Testing the model...")
    test(model, test_loader, criterion)
