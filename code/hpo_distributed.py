import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects.double() / len(test_loader.dataset)

    logger.info(f"Testing Loss: {total_loss:.4f}")
    logger.info(f"Testing Accuracy: {total_acc:.4f}")

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    epochs = 50
    best_loss = float('inf')
    loss_counter = 0

    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        train_loader.sampler.set_epoch(epoch)
        for phase, loader in zip(['train', 'valid'], [train_loader, validation_loader]):
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
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

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                loss_counter = 0
            elif phase == 'valid':
                loss_counter += 1

            logger.info(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}, best loss: {best_loss:.4f}')

        if loss_counter >= 3:
            logger.info("Early stopping triggered.")
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

def create_data_loaders(data, batch_size, world_size, rank):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.ImageFolder(root=os.path.join(data, 'train'), transform=train_transform)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

    test_data = torchvision.datasets.ImageFolder(root=os.path.join(data, 'test'), transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    validation_data = torchvision.datasets.ImageFolder(root=os.path.join(data, 'valid'), transform=test_transform)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, validation_loader

def main(args):
    hosts = os.getenv("SM_CURRENT_INSTANCE_GROUP_HOSTS").strip("[]").replace('"', '').split(",")
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["MASTER_ADDR"] = hosts[0]
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="gloo", init_method="env://")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, validation_loader = create_data_loaders(args.data, args.batch_size, world_size, rank)

    model = net().to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.module.fc.parameters(), lr=args.learning_rate)

    if os.environ["SM_CURRENT_HOST"] == os.environ["MASTER_ADDR"]:
        logger.info("Starting Model Training")
    model = train(model, train_loader, validation_loader, criterion, optimizer, device)

    if os.environ["SM_CURRENT_HOST"] == os.environ["MASTER_ADDR"]:
        logger.info("Testing Model")
        test(model.module, test_loader, criterion, device)
        logger.info("Saving Model")
        torch.save(model.module.state_dict(), os.path.join(args.model_dir, "model.pth"))

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()
    main(args)
