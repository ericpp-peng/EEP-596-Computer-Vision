import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d  # TODO: use torch.nn.functional
import os

# region DEBUG
DEBUG = False


def print_debug(*args, **kwargs):
    if DEBUG:
        print(args, kwargs)


if __name__ == "__main__":
    DEBUG = os.environ.get("PYTHON_DEBUG_MODE")
    if DEBUG is not None and DEBUG.lower() == "true":
        DEBUG = True
        print("DEBUG mode is enabled")
# endregion


def load_image_in_grayscale(filepath) -> torch.tensor:
    return cv.imread(filepath, cv.IMREAD_GRAYSCALE)


def sum_of_abs_diff(nparray1: np.array, nparray2: np.array) -> int:
    return (np.abs(nparray1 - nparray2)).sum().item()


def scanlines(tb_left: np.array, tb_right: np.array, max_d: int = 30):
    row_idx = 152
    col_idx1 = 102
    col_len = 100

    # left scanline: fixed window
    tb_left_cropped = tb_left[row_idx, col_idx1 : col_idx1 + col_len]

    g_best = None
    d_best = 0

    # disparity d: how much the right image is shifted to the right
    # so the corresponding point in the right image is at x - d
    for d in range(max_d + 1):
        start = col_idx1 - d
        end = start + col_len
        # avoid negative index or going out of bounds
        if start < 0 or end > tb_right.shape[1]:
            continue

        tb_right_cropped = tb_right[row_idx, start:end]
        g = sum_of_abs_diff(tb_left_cropped, tb_right_cropped)

        if g_best is None or g < g_best:
            g_best, d_best = g, d

    print("Best disparity at row 152:", d_best, "with SAD", g_best)
    return d_best


def plot_1d_array(array, title, xlabel=None, ylabel=None, save_image=True):
    domain = range(len(array))
    plt.plot(domain, array, marker="o")
    plt.xlabel(title)
    plt.ylabel(xlabel)
    plt.title(ylabel)
    plt.grid(True)
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def plot_2d_array_as_image(array2d: np.array, title, save_image=True):
    plt.imshow(array2d, cmap="gray")
    plt.title(title)
    plt.colorbar()
    if save_image:
        plt.savefig(f"figure/{title}.png")
    plt.show()


def shift_array(nparray: np.array, d: int) -> np.array:
    shifted = np.zeros_like(nparray)
    if d == 0:
        shifted[:, :] = nparray[:, :]
    elif d > 0:
        shifted[:, d:] = nparray[:, :-d]
    elif d < 0:
        shifted[:, : nparray.shape[1] + d] = nparray[:, -d:]
    return shifted


if DEBUG:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert (shift_array(a, 1) == [[0, 1, 2], [0, 4, 5], [0, 7, 8]]).all()
    assert (shift_array(a, 2) == [[0, 0, 1], [0, 0, 4], [0, 0, 7]]).all()


def auto_correlation(tb_right):
    max_d = 30
    auto_correlations = []
    for d in range(max_d + 1):
        abs_diff_image = np.abs(tb_right - shift_array(tb_right, d))
        auto_correlations.append(abs_diff_image[152][152])

    if DEBUG:
        plot_1d_array(auto_correlations, auto_correlation.__name__)
    return auto_correlations


def convolve2d_torch(array: np.array, kernel_size: int):
    as_tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    convolved = nn.functional.conv2d(as_tensor, kernel, padding=kernel_size // 2)
    if DEBUG:
        assert convolved.shape == as_tensor.shape

    return np.array(convolved.squeeze().squeeze())

def smoothing(tb_right, max_d: int = 30, kernel_size: int = 5):
    """
    Compute smoothed auto-correlation at pixel (152,152)
    by applying a box filter to each absolute-difference image.
    """
    smoothed_auto = []
    for d in range(max_d + 1):
        shifted = shift_array(tb_right, d)
        abs_diff_image = np.abs(tb_right - shifted)
        smoothed = convolve2d_torch(abs_diff_image, kernel_size)
        smoothed_auto.append(smoothed[152, 152])

    if DEBUG:
        plot_1d_array(smoothed_auto, "smoothed_auto_correlation",
                      xlabel="disparity d", ylabel="value")

    return smoothed_auto


# def cross_correlation(tb_left, tb_right):


# def disparity_map(


# def right_left_disparity(tb_left, tb_right, plot_result=False):



# def disparity_check(tb_left, tb_right):



# def reconstruction(tb_left, tb_right):
#     # Fill your code hear
#     pass


if __name__ == "__main__":
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")
    scanlines(tb_left, tb_right)
    auto_correlation(tb_right)
    smoothing(tb_right)
    # cross_correlation(tb_left, tb_right)
    # disparity_map(tb_left, tb_right, plot_result=True)
    # right_left_disparity(tb_left, tb_right, plot_result=True)
    # disparity_check(tb_left, tb_right)
