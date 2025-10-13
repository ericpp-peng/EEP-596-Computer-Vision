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
    kernel = 0.25 * np.array([1, 2, 1])

    # Make a copy of the original grayscale image
    image = self.cat_eye.astype(np.float32)

    # Store blurred images
    self.blurred_images = []
    
    for i in range(5):
        # Apply convolution
        # --- Vertical convolution (3x1) ---
        h, w = image.shape

        # Add one row of zeros to the top and bottom to handle boundary pixels safely
        padded = np.pad(image, ((1, 1), (0, 0)), mode='constant', constant_values=0)

        # Create an empty image (same size and type as the input image) to store the vertically blurred result
        vert_blur = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                # Multiply 3 neighboring pixels vertically with kernel
                vert_blur[y, x] = (
                    kernel[0] * padded[y, x] +
                    kernel[1] * padded[y + 1, x] +
                    kernel[2] * padded[y + 2, x]
                )
            # --- Add this block to visualize the convolution process dynamically ---
            # if y % 1 == 0:  # Update every 10 rows (you can change to 1 for smoother animation)
                # display_img = np.clip(vert_blur / np.max(vert_blur) * 255, 0, 255).astype(np.uint8)
                # cv2.imshow("Vertical Convolution Progress", display_img)
                # cv2.waitKey(1)  # Wait 1ms to create animation effect

        # --- Horizontal convolution (1x3) ---
        # Add one row of zeros to the left and right to handle boundary pixels safely
        padded_h = np.pad(vert_blur, ((0, 0), (1, 1)), mode='constant', constant_values=0)

        # Create an empty image (same size and type as vert_blur) to store the horizontally blurred result
        horiz_blur = np.zeros_like(vert_blur)

        # Vertical and horizontal convolutions differ in direction but share the same scanning order
        for y in range(h):
            for x in range(w):
                horiz_blur[y, x] = (
                    kernel[0] * padded_h[y, x] +
                    kernel[1] * padded_h[y, x + 1] +
                    kernel[2] * padded_h[y, x + 2]
                )

        # Clamp to [0, 255] and convert to uint8
        horiz_blur = np.clip(np.round(horiz_blur), 0, 255).astype(np.uint8)

        # Save the current blurred image
        self.blurred_images.append(horiz_blur)

        # Update the image for the next iteration
        image = horiz_blur.astype(np.float32)

        # save the image
        # cv2.imwrite(f"task_outputs/Task2_blur_{i}.jpg", horiz_blur)

    return self.blurred_images

  def gaussian_derivative_vertical(self):

    # 1D kernels
    ks = 0.25 * np.array([1, 2, 1], dtype=np.float32)   # horizontal smoothing (1×3)
    kd_v = 0.5  * np.array([-1, 0, 1], dtype=np.float32)    # derivative (flipped for convolution)

    # Store images
    self.vDerive_images = []

    for i in range(5):
      img = self.blurred_images[i].astype(np.float32)
      h, w = img.shape

      # Horizontal smoothing with zero padding on left/right
      padded_h = np.pad(img, ((0, 0), (1, 1)), mode='constant', constant_values=0)
      hsm = np.zeros_like(img, dtype=np.float32)

      for y in range(h):
        for x in range(w):
          hsm[y, x] = (
            ks[0] * padded_h[y, x] +
            ks[1] * padded_h[y, x+1] +
            ks[2] * padded_h[y, x+2]
          )

      # Vertical derivative with zero padding on top/bottom
      # Note the reversed indexing (y+2, y+1, y) to emulate convolution with the flipped kernel
      padded_v = np.pad(hsm, ((1, 1), (0, 0)), mode='constant', constant_values=0)
      vresp = np.zeros_like(hsm, dtype=np.float32)

      for y in range(h):
        for x in range(w):
          vresp[y, x] = (
            kd_v[0] * padded_v[y+2, x] +
            kd_v[1] * padded_v[y+1, x] +
            kd_v[2] * padded_v[y  , x]
          )

      # Convert to uint8: pout = clamp(2 * pin + 127)
      pout = 2.0 * vresp + 127.0
      pout = np.clip(np.round(pout), 0, 255).astype(np.uint8)

      self.vDerive_images.append(pout)
      # cv2.imwrite(f"task_outputs/Task3_vertical_derivative_{i}.jpg", pout)

    return self.vDerive_images

  def gaussian_derivative_horizontal(self):

    ks   = 0.25 * np.array([1, 2, 1], dtype=np.float32)   # smoothing (vertical)
    kd_h = 0.5  * np.array([-1, 0, 1], dtype=np.float32)  # derivative (flipped for convolution)

    self.hDerive_images = []

    for i in range(5):
      img = self.blurred_images[i].astype(np.float32)
      h, w = img.shape

      # Vertical smoothing with zero padding on top/bottom
      padded_v = np.pad(img, ((1, 1), (0, 0)), mode='constant', constant_values=0)
      vsm = np.zeros_like(img, dtype=np.float32)
      for y in range(h):
        for x in range(w):
          vsm[y, x] = (
            ks[0] * padded_v[y,   x] +
            ks[1] * padded_v[y+1, x] +
            ks[2] * padded_v[y+2, x]
          )

      # Horizontal derivative with zero padding on left/right
      # Reversed indexing (x+2, x+1, x) to emulate convolution with the flipped kernel
      padded_h = np.pad(vsm, ((0, 0), (1, 1)), mode='constant', constant_values=0)
      hresp = np.zeros_like(vsm, dtype=np.float32)
      for y in range(h):
        for x in range(w):
          hresp[y, x] = (
            kd_h[0] * padded_h[y, x+2] +
            kd_h[1] * padded_h[y, x+1] +
            kd_h[2] * padded_h[y, x]
          )

      # Convert to uint8: pout = clamp(2 * pin + 127)
      pout = 2.0 * hresp + 127.0
      pout = np.clip(np.round(pout), 0, 255).astype(np.uint8)

      self.hDerive_images.append(pout)
      # cv2.imwrite(f"task_outputs/Task4_horizontal_derivative_{i}.jpg", pout)

    return self.hDerive_images

  def gradient_magnitute(self):

    # 1D kernels
    ks   = 0.25 * np.array([1, 2, 1], dtype=np.float32)   # smoothing (symmetric)
    kder = 0.5  * np.array([-1, 0, 1], dtype=np.float32)  # derivative (flipped for convolution)

    # reset list to store the 5 gradient-magnitude images (one per blur level, uint8)
    self.gdmagnitude_images = []

    for i in range(5):
        img = self.blurred_images[i].astype(np.float32)
        h, w = img.shape

        # gx: vertical smoothing (3×1)
        pad_v = np.pad(img, ((1, 1), (0, 0)), mode='constant', constant_values=0)
        vsm = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                vsm[y, x] = (
                    ks[0] * pad_v[y,     x] +
                    ks[1] * pad_v[y + 1, x] +
                    ks[2] * pad_v[y + 2, x]
                )

        # gx: horizontal derivative (1×3), reversed indexing for convolution
        pad_h = np.pad(vsm, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        gx = np.zeros_like(vsm, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                gx[y, x] = (
                    kder[0] * pad_h[y, x + 2] +
                    kder[1] * pad_h[y, x + 1] +
                    kder[2] * pad_h[y, x    ]
                )
        gx = np.abs(gx)

        # gy: horizontal smoothing (1×3)
        pad_h2 = np.pad(img, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        hsm = np.zeros_like(img, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                hsm[y, x] = (
                    ks[0] * pad_h2[y, x    ] +
                    ks[1] * pad_h2[y, x + 1] +
                    ks[2] * pad_h2[y, x + 2]
                )

        # gy: vertical derivative (3×1), reversed indexing for convolution
        pad_v2 = np.pad(hsm, ((1, 1), (0, 0)), mode='constant', constant_values=0)
        gy = np.zeros_like(hsm, dtype=np.float32)
        for y in range(h):
            for x in range(w):
                gy[y, x] = (
                    kder[0] * pad_v2[y + 2, x] +
                    kder[1] * pad_v2[y + 1, x] +
                    kder[2] * pad_v2[y,     x]
                )
        gy = np.abs(gy)

        # Manhattan magnitude and uint8 conversion
        pin = gx + gy
        pout = 4.0 * pin
        pout = np.clip(np.round(pout), 0, 255).astype(np.uint8)

        self.gdmagnitude_images.append(pout)
        # cv2.imwrite(f"task_outputs/Task5_gradmag_{i}.jpg", pout)

    return self.gdmagnitude_images
    
  def scipy_convolve(self):
    
    # 1D kernels: smoothing is symmetric; derivative is flipped for true convolution
    ks_row = (0.25 * np.array([1, 2, 1], dtype=np.float32)).reshape(1, 3)   # 1×3
    kder_col = (0.5  * np.array([1, 0, -1], dtype=np.float32)).reshape(3, 1) # 3×1 (vertical derivative; unflipped)

    self.scipy_smooth = []

    for i in range(5):
        img = self.blurred_images[i].astype(np.float32)

        # Step 1: horizontal smoothing (1×3), zero padding, same size
        smoothed = scipy.signal.convolve2d(
            img, ks_row, mode='same', boundary='fill', fillvalue=0
        )

        # Step 2: vertical derivative (3×1), zero padding, same size
        vresp = scipy.signal.convolve2d(
            smoothed, kder_col, mode='same', boundary='fill', fillvalue=0
        )

        # Map to uint8 for visualization (same as Task 3)
        pout = 2.0 * vresp + 127.0
        pout = np.clip(np.round(pout), 0, 255).astype(np.uint8)

        self.scipy_smooth.append(pout)
        # cv2.imwrite(f"task_outputs/Task6_scipy_vertical_derivative_{i}.jpg", pout)

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
