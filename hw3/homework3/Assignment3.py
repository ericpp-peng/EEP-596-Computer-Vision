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
        # Convert BGR -> RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Convert NumPy array to torch tensor (float64) and rescale pixel values to [0, 1].
        # CNNs typically expect input images normalized to this range for stable training and inference.
        x = torch.from_numpy(img_rgb).to(torch.float64) / 255.0

        # Clamp is not strictly necessary here (values should already be in [0,1]),
        # but it helps guard against rare rounding errors or unexpected upstream inputs.
        x = x.clamp(0.0, 1.0)

        # Normalize image per channel (R, G, B) so that each channel has mean ≈ 0 and std ≈ 1
        # Compute the mean value for each channel (R, G, B)
        # Take the average over the height (H) and width (W) dimensions
        # Result shape: [C], where C = 3 for RGB
        mean = x.mean(dim=(0, 1))

        # Compute the standard deviation (spread of values) for each channel
        # Use population standard deviation (unbiased=False) because we treat the whole image as a full population
        std  = x.std(dim=(0, 1), unbiased=False)

        # Small epsilon to prevent division by zero when std is extremely small
        eps = 1e-12

        # Normalize each pixel value:
        #   1. Subtract the mean → centers values around 0 (removes brightness bias)
        #   2. Divide by std → scales values to have unit variance (removes contrast bias)
        # The result z has mean ≈ 0 and std ≈ 1 for each channel
        z = (x - mean) / (std + eps)


        # Ensure normalized values are within [-1, 1] range
        # This prevents extreme pixel values from causing instability
        # and matches the expected input range for CNN model.
        z = z.clamp(-1.0, 1.0)

        return z

    def Imagenet_norm(self, img):
        # Convert BGR (OpenCV default) to RGB (used by PyTorch/ImageNet)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Convert to float64 and scale pixel values from [0, 255] → [0, 1]
        x = torch.from_numpy(img_rgb).to(torch.float64) / 255.0
        x = x.clamp(0.0, 1.0)

        # ImageNet per-channel mean and std (for RGB)
        im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float64)
        im_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float64)

        # Normalize each channel to match ImageNet training distribution
        z = (x - im_mean) / im_std

        # Clamp to [-1, 1] to match the expected input range of the model
        z = z.clamp(-1.0, 1.0)

        return z

    def dimension_rearrange(self, img):
        # Load image in BGR (OpenCV default) and convert to RGB
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Convert NumPy array to torch tensor (float32)
        x = torch.from_numpy(img_rgb).to(torch.float32)

        # Rearrange dimensions:
        # Original: (H, W, C)
        # Target:   (N, C, H, W) where N=1 for a single image
        x = x.permute(2, 0, 1).unsqueeze(0)

        return x
    

    def stride(self, img):
        # Step 1. img is already grayscale (H, W)
        x = torch.from_numpy(img).to(torch.float32)

        # Step 2. Define 3x3 Scharr_x kernel
        scharr_x = torch.tensor([
            [-3.,  0.,  3.],
            [-10., 0., 10.],
            [-3.,  0.,  3.]
        ], dtype=torch.float32)

        # Step 3. Flip kernel for true convolution
        scharr_x = torch.flip(scharr_x, dims=[0, 1])

        # Step 4. Pad image with 0 (pad=1 keeps size for stride=1)
        x_padded = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode='constant', value=0)

        # Step 5. Manual stride convolution
        stride = 2
        H, W = x_padded.shape
        out_H = (H - 3) // stride + 1
        out_W = (W - 3) // stride + 1
        y = torch.zeros((out_H, out_W), dtype=torch.float32)

        for i in range(0, H - 2, stride):
            for j in range(0, W - 2, stride):
                region = x_padded[i:i+3, j:j+3]
                y[i // stride, j // stride] = torch.sum(region * scharr_x)

        # Return 2D FloatTensor
        return y

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
    stride_img = assign.stride(img)
    df_dx, df_dy, df_dz, df_dq = assign.chain_rule(x=-2.0, y=5.0, z=-4.0)
    dx, dw = assign.relu(x=[-1.0, 2.0], w=[2.0, -3.0, -3.0])
