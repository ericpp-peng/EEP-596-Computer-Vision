# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (e.g., bright green)
    fill_color =   # (B, G, R)
    # Create a copy of the input image to keep the original image unchanged
    output_image =
    # Define a stack for floodfill

    #cv2.imwrite('floodfille.jpg', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = # 1D Gaussian kernel
    image = self.cat_eye
    self.blurred_images = []
    for i in range(5):
        # Apply convolution
        image=
        
        # Store the blurred image
        self.blurred_images.append(image)
        
        #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution
      image =
      
      self.vDerive_images.append(image)
      #cv2.imwrite(f'vertical {i}.jpg', image)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels

    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):

      # Apply horizontal and vertical convolution
      image =

      self.hDerive_images.append(image)
      #cv2.imwrite(f'horizontal {i}.jpg', image)
    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
      image = 
      self.gdMagnitute_images.append(image)
      #cv2.imwrite(f'gradient {i}.jpg', image)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
   
    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution
      image=
      self.scipy_smooth.append(image)
      #cv2.imwrite(f'scipy smooth {i}.jpg', image)
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D conlve
      out =

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    # floodfill_img = ass.floodfill(100, 100)

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
