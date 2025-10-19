import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, img):
        # BGR → RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # OpenCV → NumPy → PyTorch Tensor
        torch_img = torch.from_numpy(img_rgb).to(torch.float32)

        return torch_img

    def brighten(self, torch_img):
        bright_img = torch_img + 100.0
        return bright_img

    def saturation_arithmetic(self, img):
        # Convert BGR to RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Convert NumPy array to torch tensor (uint8)
        torch_img = torch.from_numpy(img_rgb)

        # Add 100 using uint8 addition (wrap-around behavior)
        # Note:
        # As clarified by TA Zhihai Zhou (Oct 2025),
        # since the prompt asks to convert the image to a uint8 tensor *before* adding 100,
        # the addition will follow uint8 overflow behavior (wrap-around),
        # e.g., 250 + 10 = 4, instead of clamping to 255.
        #
        # This matches the autograder’s reference solution.
        # However, keep in mind that "saturation arithmetic"
        # in general means clamping values at 255 — a key distinction to remember for future work.
        saturated_img = torch_img + torch.tensor(100, dtype=torch.uint8)

        return saturated_img

    def add_noise(self, torch_img):
        # Add random Gaussian noise with mean=0 and std=100.0 (in pixel intensity units)
        mean = 0.0
        std = 100.0

        # Convert to float32 for arithmetic (only changes dtype; values stay in 0..255)
        x = torch_img.to(torch.float32)

        # Generate Gaussian noise N(mean, std^2) with the same shape as the image
        noise = torch.randn_like(x) * std + mean

        # Add noise and clamp to [0, 255] to ensure valid pixel intensity range
        y = (x + noise).clamp(0.0, 255.0)

        # Normalize pixel values to [0, 1]; clamp again for safety
        noisy_img = (y / 255.0).clamp(0.0, 1.0).to(torch.float32)

        return noisy_img

    def normalization_image(self, img):

        return image_norm

    def Imagenet_norm(self, img):

        return ImageNet_norm

    def dimension_rearrange(self, img):

        return rearrange

    def chain_rule(self, x, y, z):

        return df_dx, df_dy, df_dz, df_dq

    def relu(self, x, w):

        return dx, dw


if __name__ == "__main__":
    img = cv.imread("original_image.png")
    assign = Assignment3()
    torch_img = assign.torch_image_conversion(img)
    bright_img = assign.brighten(torch_img)
    saturated_img = assign.saturation_arithmetic(img)
    noisy_img = assign.add_noise(torch_img)
    image_norm = assign.normalization_image(img)
    ImageNet_norm = assign.Imagenet_norm(img)
    rearrange = assign.dimension_rearrange(img)
    df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
