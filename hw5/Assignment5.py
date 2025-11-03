import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def chain_rule():
    """
    Compute df/dz, df/dq, df/dx, and df/dy for f(x,y,z)=xy+z,
    where q=xy, at x=-2, y=5, z=-4.
    Return them in this order: df/dz, df/dq, df/dx, df/dy. 
    """ 
    return 

def ReLU():
    """
    Compute dx and dw, and return them in order.
    Forward:
        y = ReLU(w0 * x0 + w1 * x1 + w2)

    Returns:
        dx -- gradient with respect to input x, as a vector [dx0, dx1]
        dw -- gradient with respect to weights (including the third term w2), 
              as a vector [dw0, dw1, dw2]
    """
    return dx, dw

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    return a, b, c

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """
    return gw0, gx0, gw1, gx1, gw2

def backprop_a():
    """
    Let f(w,x) = torch.tanh(w0x0+w1x1+w2).  
    Assume the weight vector is w = [w0=5, w1=2], 
    the input vector is  x = [x0=-1,x1= 4],, and the bias is  w2  =-2.
    Use PyTorch to calculate the forward pass of the network, return y_hat = f(w,x).
    """
    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """

    return gw0, gw1, gw2

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    return  w0, w1, w2 


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def newtonMethod(x0, y0):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0) 
    paraboloid = torch.unsqueeze(paraboloid, 0)    # -> (1,1,H,W) for conv2d

    """
    Insert your code here
    """

    return final_x, final_y


def sgd(x0, y0, lr=0.001):
    paraboloid = torch.tensor([constructParaboloid()]).squeeze()
    paraboloid = torch.unsqueeze(paraboloid, 0)
    paraboloid = torch.unsqueeze(paraboloid, 0)

    """
    Insert your code here
    """

    return final_x, final_y


