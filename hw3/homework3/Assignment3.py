import numpy as np
import torch
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt


class Assignment3:
    def __init__(self) -> None:
        pass

    def torch_image_conversion(self, torch_img):

        return torch_img

    def brighten(self, torch_img):

        return bright_img

    def saturation_arithmetic(self, img):

        return saturated_img

    def add_noise(self, torch_img):

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
