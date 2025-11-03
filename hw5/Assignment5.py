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
    x, y, z = -2.0, 5.0, -4.0
    df_dz = 1.0
    df_dq = 1.0
    df_dx = y  # 5
    df_dy = x  # -2
    return df_dz, df_dq, df_dx, df_dy

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
    # given data
    x0, x1 = -1.0, -2.0
    w0, w1, w2 = 2.0, -3.0, -3.0

    s = w0 * x0 + w1 * x1 + w2  # pre-activation
    if s > 0:
        dy_ds = 1.0
        dx0 = dy_ds * w0
        dx1 = dy_ds * w1
        dw0 = dy_ds * x0
        dw1 = dy_ds * x1
        dw2 = dy_ds * 1.0
    else:
        dx0 = dx1 = 0.0
        dw0 = dw1 = dw2 = 0.0

    dx = [dx0, dx1]
    dw = [dw0, dw1, dw2]
    return dx, dw

def chain_rule_a():
    """
    In the lecture notes, the last three forward pass values are 
    a=0.37, b=1.37, and c=0.73.  
    Calculate these numbers to 4 decimal digits and return in order of a, b, c
    """
    
    a = torch.exp(torch.tensor(-1.0))
    b = 1 + a
    c = 1 / b
    # Round to 4 decimal digits
    return round(a.item(), 4), round(b.item(), 4), round(c.item(), 4)

def chain_rule_b():
    """
    In the lecture notes, the backward pass values are
    ±0.20, ±0.39, -0.59, and -0.53.  
    Calculate these numbers to 4 decimal digits 
    and return in order of gradients for w0, x0, w1, x1, w2.
    """

    w0 = torch.tensor(2.0, requires_grad=True)
    x0 = torch.tensor(-1.0, requires_grad=True)
    w1 = torch.tensor(-3.0, requires_grad=True)
    x1 = torch.tensor(-2.0, requires_grad=True)
    w2 = torch.tensor(-3.0, requires_grad=True)

    # forward pass
    a = w0 * x0 + w1 * x1 + w2
    f = 1 / (1 + torch.exp(-a))

    # backward pass
    f.backward()

    # keep tensor type but round to 4 decimals
    def r(t):
        return torch.tensor(round(t.item(), 4))

    return r(w0.grad), r(x0.grad), r(w1.grad), r(x1.grad), r(w2.grad)

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


