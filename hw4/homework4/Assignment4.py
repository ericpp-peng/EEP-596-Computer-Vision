import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    return images, labels

class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # Inits the model layers

    def forward(self, x):
        # Defines forward apth
        return x

def train_classifier():
    # Creates dataset

    # Creates Network 
    net = Net()


    # Defines loss function and optimizer

    for epoch in range(2):  # loop over the dataset for 2 iteration
        pass

    # Saves the model weights after training
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # Loads dataset
    batch_size=4
    transform = 
    testset = 
    testloader =
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # Evaluates samples

def get_first_layer_weights():
    net = Net()
    # TODO: load the trained weights
    first_weight = None  # TODO: get conv1 weights (exclude bias)
    return first_weight

def get_second_layer_weights():
    net = Net()
    # TODO: load the trained weights
    second_weight = None  # TODO: get conv2 weights (exclude bias)
    return second_weight

def hyperparameter_sweep():
    '''
    Reuse the CNN and training code from Question 2
    Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
    During training, record the training loss every 2000 iterations
    compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
    After training, plot three curves
    '''
    return None

if __name__ == "__main__":
    # your text code here