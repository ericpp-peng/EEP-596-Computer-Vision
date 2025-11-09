import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np


def compute_num_parameters(net:nn.Module):
    """compute the number of trainable parameters in *net* e.g., ResNet-34.  
    Return the estimated number of parameters Q1. 
    """
    num_para =
    return num_para


def CIFAR10_dataset_a():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./cifar10/", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    return images, labels


class GAPNet(nn.Module):
    """
    Insert your code here
    """


def train_GAPNet():
    """
    Insert your code here
    """


def eval_GAPNet():
    """
    Insert your code here
    """

def backbone():
    """
    Insert your code here, Q3
    """
    return features

def transfer_learning():
    """
    Insert your code here, Q4
    """

class MobileNetV1(nn.Module):
    """Define MobileNetV1 please keep the strucutre of the class Q5"""
    def __init__(self, ch_in, n_classes):


    def forward(self, x):

    
if __name__ == '__main__':
    #Q1
    from torchvision import models
    resnet34 = models.resnet34(pretrained=True)
    num_para = compute_num_parameters(resnet34)
    # Q5
    ch_in=3
    n_classes=1000
    model = MobileNetV1(ch_in=ch_in, n_classes=n_classes)
