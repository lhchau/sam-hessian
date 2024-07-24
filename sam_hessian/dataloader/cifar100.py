import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from .cutout import Cutout


def get_cifar100(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2) 
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    dataset = torchvision.datasets.CIFAR100(root=os.path.join('.', 'data'), train=True, download=True, transform=transform_train)
    labels = np.array(dataset.targets)
    _, val_size = split

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    for train_index, val_index in stratified_split.split(np.zeros(len(labels)), labels):
        data_train = Subset(dataset, train_index)
        data_val = Subset(dataset, val_index)
    
    data_test = torchvision.datasets.CIFAR100(root=os.path.join('.', 'data'), train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader, len(data_test.classes)