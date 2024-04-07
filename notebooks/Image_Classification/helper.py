import torch.nn as nn
import os
import torch.nn.functional as F
from collections import Counter
from torchvision import transforms, datasets
import torchvision
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import time
from tqdm import tqdm


def set_device():
    """
    Set the device to use for training and inference.
    """
    print(f"PyTorch version: {torch.__version__}")

    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device      
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    return device

def set_deterministic():
    """
    Set deterministic behavior for reproducibility.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # Currently, PyTorch-Metal (MPS backend) does not provide a direct way to set deterministic behavior.
        pass
    print("Set deterministic behavior")

    
def set_seed(seed):
    """
    Set seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Set seed for reproducibility: {seed}")
    
    
    
def build_dataloaders_cifar(data_path, train_transforms, test_transforms,  num_workers, batch_size): 
    """
    Build dataloaders for CIFAR-10 dataset.
    """
    
    train_dataset = datasets.CIFAR10(data_path, 
                                    train=True,
                                    download=True,
                                    transform=train_transforms)
    
    test_dataset = datasets.CIFAR10(data_path,
                                    train=False,
                                    download=True,
                                    transform=test_transforms)
    
    # Load the dataset for visualization without any transforms
    test_dataset_viz = datasets.CIFAR10(data_path,
                                    train=False,
                                    download=True)
  
    
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers, 
                                persistent_workers=True)
    
    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                persistent_workers=True)
    
    class_names = train_dataset.classes
    
    
    return train_dataloader, test_dataloader, test_dataset_viz, class_names



def build_dataloaders_mnist(data_path, train_transforms, test_transforms,  num_workers, batch_size):
    """
    Build dataloaders for MNIST dataset.
    """
    
    train_dataset = datasets.MNIST(data_path,
                                    train=True,
                                    download=True,
                                    transform=train_transforms)
    
    test_dataset = datasets.MNIST(data_path,
                                    train=False,
                                    download=True,
                                    transform=test_transforms)
    
    # Load the dataset for visualization without any transforms
    test_dataset_viz = datasets.MNIST(data_path,
                                    train=False,
                                    download=True)
    
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers, 
                                persistent_workers=True)
    
    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                persistent_workers=True)
    
    class_names = train_dataset.classes
    
    return train_dataloader, test_dataloader, test_dataset_viz, class_names
    

def build_dataloaders_fmnist(data_path, train_transforms, test_transforms, num_workers, batch_size): 
    """
    Function to build dataloaders for the Fashion MNIST dataset
    
    """
    
    train_dataset = datasets.FashionMNIST(data_path, 
                                          train=True, 
                                          download=True, 
                                          transform=train_transforms, 
                                         )
    
    test_dataset = datasets.FashionMNIST(data_path, 
                                         train=False, 
                                         download=True, 
                                         transform=test_transforms)
    
    test_dataset_viz = datasets.FashionMNIST(data_path, 
                                             train=False, 
                                             download=True, 
                                             transform=transforms.Compose([transforms.ToTensor()]))
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  num_workers=num_workers, 
                                  persistent_workers=True)
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False, 
                                 num_workers=num_workers, 
                                 persistent_workers=True)
    
    class_names = train_dataset.classes
    
    return train_dataloader, test_dataloader, test_dataset_viz, class_names

    
    
def train_model(model, train_dataloader, test_dataloader, epochs, criterion, optimizer, device, scheduler = None):
    """
    This function trains the model
    
    Parameters
    ----------
    model: nn.Module
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    epochs: int
    criterion: nn.Module
    optimizer: nn.Module
    device: str
    """
    
    model.to(device)
    
    results = {"train_loss_per_batch": [],
               "train_loss_per_epoch": [],
               "train_acc_per_epoch": [], 
               "val_loss_per_epoch": [], 
               "val_acc_per_epoch": []}
    
    for epoch in tqdm(range(epochs)):
        model.train()
        training_loss, training_acc = 0.0, 0.0
        
        for batch, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            results['train_loss_per_batch'].append(loss.item())
            training_loss += loss.item()
            training_acc += (outputs.argmax(1) == labels).float().mean()
            training_acc = training_acc.item()
            
        scheduler.step()
        epoch_loss_tr = training_loss / len(train_dataloader)
        epoch_acc_tr = training_acc / len(train_dataloader)
        
        print(f'Epoch: {epoch} Training Loss: {epoch_loss_tr}, Training Accuracy: {epoch_acc_tr}')
        
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        
        for batch, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            test_acc += (outputs.argmax(1) == labels).float().mean()
            test_acc = test_acc.item()
            
        epoch_loss_te = test_loss / len(test_dataloader)
        epoch_acc_te = test_acc / len(test_dataloader)
        
        print(f'Epoch: {epoch} Validation Loss: {epoch_loss_te}, Validation Accuracy: {epoch_acc_te}')
    
        results['train_loss_per_epoch'].append(epoch_loss_tr)
        results['train_acc_per_epoch'].append(epoch_acc_tr)
        results['val_loss_per_epoch'].append(epoch_loss_te)
        results['val_acc_per_epoch'].append(epoch_acc_te)
        
    return model, results


def evaluate_model(pytorch_model, test_dataloader, device, class_names):
    y_preds = []

    pytorch_model.eval()
    with torch.inference_mode(): 
        for X, y in tqdm(test_dataloader): 
            X, y = X.to(device), y.to(device)
            logits = pytorch_model(X)
            pred = torch.argmax(logits, dim = 1)
            pred = pred.cpu()
            
            y_preds.append(pred)
            
    y_preds = torch.cat(y_preds, dim = 0)
    test_truth = torch.cat([y for _, y in test_dataloader], dim = 0 )

    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass').to(device)
    confmat_tensor = confmat(y_preds.to(device), test_truth.to(device))
    confmat_tensor = confmat_tensor.cpu()

    fig, ax = plot_confusion_matrix(confmat_tensor.numpy(), figsize=(10, 10), class_names=class_names, show_normed=True)
    fig.show()
    
    return y_preds, test_truth