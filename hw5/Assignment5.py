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
    # Define weights and inputs as float tensors (no gradient needed yet)
    w0 = torch.tensor(5.0)
    w1 = torch.tensor(2.0)
    w2 = torch.tensor(-2.0)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)

    # Compute the linear combination
    z = w0 * x0 + w1 * x1 + w2
    # Apply the tanh activation function (must use torch.tanh for autograd support)
    y_hat = torch.tanh(z)

    # Return scalar float value (not tensor)
    return y_hat

def backprop_b():
    """
    Use PyTorch Autograd to calculate the gradients 
    for each of the weights, and return the gradient of them 
    in order of w0, w1, and w2.
    """
    # Define inputs and weights, all require gradients
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    target = torch.tensor(1.0)  # ground truth

    # Forward pass
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)

    # Define MSE loss
    loss = (y_hat - target) ** 2

    # Backward pass (compute gradients)
    loss.backward()

    # Return gradients as tensors
    return w0.grad, w1.grad, w2.grad

def backprop_c():
    """
    Assuming a learning rate of 0.1, 
    update each of the weights accordingly. 
    For simplicity, just do one iteration. 
    And return the updated weights in the order of w0, w1, and w2 
    """
    # Define inputs and weights, all require gradients
    w0 = torch.tensor(5.0, requires_grad=True)
    w1 = torch.tensor(2.0, requires_grad=True)
    w2 = torch.tensor(-2.0, requires_grad=True)
    x0 = torch.tensor(-1.0)
    x1 = torch.tensor(4.0)
    target = torch.tensor(1.0)

    # Forward pass
    z = w0 * x0 + w1 * x1 + w2
    y_hat = torch.tanh(z)

    # MSE loss
    loss = (y_hat - target) ** 2

    # Compute gradients
    loss.backward()

    # Learning rate
    lr = 0.1

    # Manual gradient descent update
    new_w0 = w0 - lr * w0.grad
    new_w1 = w1 - lr * w1.grad
    new_w2 = w2 - lr * w2.grad

    # Return updated weights as torch.Tensors
    return new_w0, new_w1, new_w2


def constructParaboloid(w=256, h=256):
    img = np.zeros((w, h), np.float32)
    for x in range(w):
        for y in range(h):
            # let's center the paraboloid in the img
            img[y, x] = (x - w / 2) ** 2 + (y - h / 2) ** 2
    return img


def newtonMethod(x0, y0):
    """
    (5a) Implement Newton's method to find the minimum of a 2D paraboloid image.
    Use image convolutions to compute first- and second-order derivatives.
    Start from (x0, y0) and iteratively apply:
        [x, y] = [x, y] - H^{-1} * grad
    Return the final (x, y) coordinates where Newton’s method converges.
    """
    # Construct paraboloid image and reshape to (1,1,H,W) for conv2d
    paraboloid = torch.tensor(constructParaboloid(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    _, _, H, W = paraboloid.shape

    max_iters = 20
    tol = 1e-6

    # Sobel kernels for first derivatives (scaled and sign-corrected)
    kx = (-0.125 * torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
    ky = (-0.125 * torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32)).unsqueeze(0).unsqueeze(0)

    # Second-derivative kernels (central difference)
    kxx = torch.tensor([[0, 0, 0],
                        [1, -2, 1],
                        [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kyy = torch.tensor([[0, 1, 0],
                        [0, -2, 0],
                        [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Derivative maps
    fx  = F.conv2d(paraboloid, kx, padding=1)
    fy  = F.conv2d(paraboloid, ky, padding=1)
    fxx = F.conv2d(paraboloid, kxx, padding=1)
    fyy = F.conv2d(paraboloid, kyy, padding=1)
    fxy = F.conv2d(F.conv2d(paraboloid, kx, padding=1), ky, padding=1)

    # Nearest-neighbor sampling helper
    def sample(field, x, y):
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(xi, W - 1))
        yi = max(0, min(yi, H - 1))
        return float(field[0, 0, yi, xi].item())

    # Iterative Newton updates
    x, y = float(x0), float(y0)
    for i in range(max_iters):
        gx, gy = sample(fx, x, y), sample(fy, x, y)
        hxx, hyy, hxy = sample(fxx, x, y), sample(fyy, x, y), sample(fxy, x, y)

        Hmat = torch.tensor([[hxx, hxy],
                             [hxy, hyy]], dtype=torch.float32)
        if torch.linalg.det(Hmat).abs().item() < 1e-8:
            continue

        g = torch.tensor([gx, gy], dtype=torch.float32)
        delta = torch.linalg.solve(Hmat, g)

        # Newton update
        x -= float(delta[0].item())
        y -= float(delta[1].item())

        # Stop if movement is very small
        if i > 0 and delta.norm().item() < tol:
            break

    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    return float(x), float(y)


def sgd(x0, y0, lr=0.05):
    """
    (5b) Implement SGD (mini-batch gradient descent) to find the minimum
    of the same paraboloid image using image gradients from convolution.
    Start from (x0, y0) and iteratively update:
        (x, y) <- (x, y) - lr * grad(x, y)
    Return the final coordinates (final_x, final_y).
    """
    paraboloid = torch.tensor(constructParaboloid(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    _, _, H, W = paraboloid.shape

    # Sobel filters for first derivatives
    kx = (-0.125 * torch.tensor([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
    ky = (-0.125 * torch.tensor([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]], dtype=torch.float32)).unsqueeze(0).unsqueeze(0)

    fx = F.conv2d(paraboloid, kx, padding=1)
    fy = F.conv2d(paraboloid, ky, padding=1)

    def sample(field, x, y):
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(xi, W - 1))
        yi = max(0, min(yi, H - 1))
        return float(field[0, 0, yi, xi].item())

    x, y = float(x0), float(y0)
    max_iters = 500
    tol = 1e-3

    for _ in range(max_iters):
        gx, gy = sample(fx, x, y), sample(fy, x, y)
        new_x = x - lr * gx
        new_y = y - lr * gy

        if ((new_x - x)**2 + (new_y - y)**2)**0.5 < tol:
            x, y = new_x, new_y
            break

        x, y = new_x, new_y

    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    return float(x), float(y)


# local debug
if __name__ == "__main__":
    print("Testing newtonMethod...")
    result = newtonMethod(10, 200)
    print("Result:", result)

    print("Testing SGD...")
    result_sgd = sgd(10, 200, lr=0.05)
    print("SGD Result:", result_sgd)
