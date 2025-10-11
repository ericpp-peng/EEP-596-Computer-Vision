# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import os

class ComputerVisionAssignment():
  def __init__(self) -> None:
    # Load input images
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color in BGR format (green)
    fill_color = (0, 0, 255)
    
    # Make a copy of the image to keep the original unchanged
    output_image = self.ant_img.copy()
    h, w = output_image.shape[:2]

    # Get the color at the seed point
    start_color = output_image[seed[1], seed[0]].tolist()

    # If the starting color is already the same as the fill color, return directly
    if start_color == list(fill_color):
        return output_image

    # Initialize stack with the seed point
    stack = [seed]

    # DFS algorithm
    # Perform stack-based (non-recursive) flood fill 
    while stack:
        x, y = stack.pop()

        # Skip if the pixel is outside image boundaries
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        # Fill only if this pixel matches the original (seed) color
        if np.array_equal(output_image[y, x], start_color):
            # Fill the pixel with the new color
            output_image[y, x] = fill_color

            # Push its 4-connected neighbors (up, down, left, right) onto the stack
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

            # Show the filling process dynamically (use a short delay for animation)
            # cv2.imshow("Flood Fill Progress", output_image)
            # cv2.waitKey(1)  # Wait 1 millisecond between updates
            # time.sleep(0.001)  # Uncomment to slow down the animation

    # save the filled image
    cv2.imwrite('task_outputs/Task1_floodfilled.jpg', output_image)

    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = np.array([1, 2, 1])  # dummy 1D Gaussian kernel
    image = self.cat_eye
    self.blurred_images = []
    for i in range(5):
        # Apply convolution (dummy: just return same image)
        image = self.cat_eye.copy()
        
        # Store the blurred image
        self.blurred_images.append(image)
        
        #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution (dummy: zeros)
      image = np.zeros_like(self.cat_eye)
      
      self.vDerive_images.append(image)
      #cv2.imwrite(f'vertical {i}.jpg', image)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels

    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):

      # Apply horizontal and vertical convolution (dummy: zeros)
      image = np.zeros_like(self.cat_eye)

      self.hDerive_images.append(image)
      #cv2.imwrite(f'horizontal {i}.jpg', image)
    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
      image = np.sqrt(np.square(vimg) + np.square(himg))  # dummy computation
      self.gdMagnitute_images.append(image)
      #cv2.imwrite(f'gradient {i}.jpg', image)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
   
    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution (dummy: same image)
      image = self.cat_eye.copy()
      self.scipy_smooth.append(image)
      #cv2.imwrite(f'scipy smooth {i}.jpg', image)
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D convolve (dummy computation)
      out = np.convolve(out, box_filter, mode='full')

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()

    # Create output directory if it doesn't exist
    os.makedirs("task_outputs", exist_ok=True)

    # # Task 1 floodfill
    floodfill_img = ass.floodfill((100, 100))

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
