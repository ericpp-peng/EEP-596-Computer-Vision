import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.signal import convolve2d  # TODO: use torch.nn.functional
import os

os.makedirs("figure", exist_ok=True)

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


def cross_correlation(tb_left, tb_right, max_d: int = 30, kernel_size: int = 5):
    """
    Compute smoothed cross-correlation at pixel (152,152)
    by comparing the LEFT image with shifted RIGHT images.
    """
    cross_corr = []

    for d in range(max_d + 1):
        shifted_right = shift_array(tb_right, d)
        abs_diff_image = np.abs(tb_left - shifted_right)
        smoothed = convolve2d_torch(abs_diff_image, kernel_size)
        cross_corr.append(smoothed[152, 152])

    if DEBUG:
        plot_1d_array(cross_corr,
                      "smoothed_cross_correlation",
                      xlabel="disparity d",
                      ylabel="abs diff")

    return cross_corr


def disparity_map(tb_left, tb_right, max_d: int = 30, kernel_size: int = 5, plot_result=False):
    """
    Compute left-to-right disparity map using smoothed cross-correlation.
    For each pixel (x,y), find the disparity d that minimizes:
        | tb_left(x,y) - shifted_right(x,y,d) | (smoothed with 5x5 filter)
    """
    H, W = tb_left.shape
    cost_volume = np.zeros((H, W, max_d + 1), dtype=np.float32)

    for d in range(max_d + 1):
        shifted_right = shift_array(tb_right, d)

        abs_diff = np.abs(tb_left.astype(np.float32) - shifted_right.astype(np.float32))
        smoothed = convolve2d_torch(abs_diff, kernel_size)

        cost_volume[:, :, d] = smoothed

    # argmin over disparity dimension
    disp = np.argmin(cost_volume, axis=2).astype(np.uint8)

    if plot_result:
        disp_vis = (disp.astype(np.float32) / max_d * 255).astype(np.uint8)
        plot_2d_array_as_image(disp_vis, "disparity_left_to_right")

    return disp



def right_left_disparity(tb_left, tb_right, max_d: int = 30, kernel_size: int = 5, plot_result=False):
    """
    Compute right-to-left disparity map.
    For each pixel (x,y) in the RIGHT image, find disparity d such that:
        right(x,y) ≈ left(x + d, y)
    Implementation:
        - Shift LEFT image to the LEFT by d (i.e., shift_array(left, -d))
        - Compute | right - shifted_left |
        - Apply 5x5 smoothing
        - Argmin over d
        - Return disparity as NEGATIVE values (for Q8 consistency check)
    """
    H, W = tb_right.shape
    cost_volume = np.zeros((H, W, max_d + 1), dtype=np.float32)

    for d in range(max_d + 1):
        # Shift left image to the LEFT by d
        shifted_left = shift_array(tb_left, d)

        abs_diff = np.abs(tb_right.astype(np.float32) - shifted_left.astype(np.float32))
        smoothed = convolve2d_torch(abs_diff, kernel_size)

        cost_volume[:, :, d] = smoothed

    # argmin returns the disparity index that gives lowest cost
    best_d = np.argmin(cost_volume, axis=2).astype(np.int16)

    # IMPORTANT: right-left disparity should be NEGATIVE
    disp_R = -best_d

    if plot_result:
        disp_vis = (best_d.astype(np.float32) / max_d * 255).astype(np.uint8)
        plot_2d_array_as_image(disp_vis, "disparity_right_to_left_magnitude")

    return disp_R




def disparity_check(tb_left, tb_right, max_d: int = 30, kernel_size: int = 5, plot_result=False):
    """
    Left-Right consistency check.
    Keep disparity dL(x,y) only if dR(x-d, y) = -dL(x,y).
    Otherwise set the disparity to 0.
    """
    # Compute both disparity maps
    dL = disparity_map(tb_left, tb_right, max_d=max_d, kernel_size=kernel_size, plot_result=False)
    dR = right_left_disparity(tb_left, tb_right, max_d=max_d, kernel_size=kernel_size, plot_result=False)

    H, W = dL.shape
    clean = np.zeros_like(dL, dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            d = int(dL[y, x])
            if d == 0:
                continue

            xr = x - d  # location in right image
            if xr < 0 or xr >= W:
                continue

            # The Q7 disparity is negative by design (-best_d)
            if int(dR[y, xr]) == -d:
                clean[y, x] = d

    if plot_result:
        disp_vis = (clean.astype(np.float32) / max_d * 255).astype(np.uint8)
        plot_2d_array_as_image(disp_vis, "disparity_cleaned")

    return clean




def reconstruction(tb_left, tb_right, max_d: int = 30, kernel_size: int = 5,
                   ply_filename: str = "kermit.ply"):
    """
    Use the cleaned disparity map to create a 3D point cloud and save as a PLY file.
    Each valid pixel becomes one vertex: x y z r g b.
    """
    # 1. Get cleaned disparity map
    clean_disp = disparity_check(tb_left, tb_right, max_d=max_d,
                                 kernel_size=kernel_size, plot_result=False)

    H, W = clean_disp.shape

    # 2. Load left color image for colors
    color_left = cv.imread("tsukuba_left.png", cv.IMREAD_COLOR)  # BGR
    if color_left is None:
        raise FileNotFoundError("Cannot load tsukuba_left.png for color info")
    color_left = cv.cvtColor(color_left, cv.COLOR_BGR2RGB)

    # 3. Simple camera model (orthographic-ish)
    f = 1.0   # focal length (arbitrary units)
    B = 1.0   # baseline (arbitrary units)
    cx = W / 2.0
    cy = H / 2.0

    points = []

    for y in range(H):
        for x in range(W):
            d = float(clean_disp[y, x])
            if d <= 0.0:
                continue  # skip invalid or zero disparity

            Z = f * B / d          # depth ~ 1 / disparity
            X = (x - cx) * Z
            Y = -(y - cy) * Z      # negative so that image y-axis goes upward

            r, g, b = color_left[y, x]
            points.append((X, Y, Z, int(r), int(g), int(b)))

    # 4. Write PLY file
    with open(ply_filename, "w") as f_out:
        f_out.write("ply\n")
        f_out.write("format ascii 1.0\n")
        f_out.write(f"element vertex {len(points)}\n")
        f_out.write("property float x\n")
        f_out.write("property float y\n")
        f_out.write("property float z\n")
        f_out.write("property uchar red\n")
        f_out.write("property uchar green\n")
        f_out.write("property uchar blue\n")
        f_out.write("end_header\n")

        for X, Y, Z, r, g, b in points:
            f_out.write(f"{X} {Y} {Z} {r} {g} {b}\n")

    print(f"Saved point cloud with {len(points)} points to {ply_filename}")
    return ply_filename



if __name__ == "__main__":
    tb_left = load_image_in_grayscale("tsukuba_left.png")
    tb_right = load_image_in_grayscale("tsukuba_right.png")

    os.makedirs("figure", exist_ok=True)

    # -----------------------------
    # Task 1: Rectification check
    # -----------------------------
    stacked = np.hstack((tb_left, tb_right))
    plt.imshow(stacked, cmap="gray")
    plt.title("Rectified stereo pair (left | right)")
    plt.axis("off")
    plt.savefig("figure/task1_rectified_check.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 2: Scanline SAD (row 152)
    # -----------------------------
    row = 152
    col_start = 102
    col_len = 100
    left_strip = tb_left[row, col_start:col_start + col_len]

    sad_vals = []
    best_d = None
    best_sad = None
    max_d = 30
    for d in range(max_d + 1):
        start = col_start - d
        end = start + col_len
        if start < 0 or end > tb_right.shape[1]:
            sad_vals.append(np.nan)
            continue
        right_strip = tb_right[row, start:end]
        sad = np.abs(left_strip.astype(np.float32) -
                     right_strip.astype(np.float32)).sum()
        sad_vals.append(sad)
        if best_sad is None or sad < best_sad:
            best_sad = sad
            best_d = d

    print(f"Best disparity at row 152 from SAD curve: d = {best_d}, SAD = {best_sad}")

    plt.plot(range(len(sad_vals)), sad_vals, marker="o")
    plt.xlabel("disparity d")
    plt.ylabel("SAD")
    plt.title("Scanline SAD at row 152")
    plt.grid(True)
    plt.savefig("figure/task2_scanline_sad.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 3: Auto-correlation
    # -----------------------------
    ac = auto_correlation(tb_right)
    plt.plot(range(len(ac)), ac, marker="o")
    plt.xlabel("disparity d")
    plt.ylabel("abs difference")
    plt.title("Auto-correlation at (152,152)")
    plt.grid(True)
    plt.savefig("figure/task3_auto_correlation.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 4: Smoothed auto-correlation
    # -----------------------------
    sm = smoothing(tb_right)
    plt.plot(range(len(sm)), sm, marker="o")
    plt.xlabel("disparity d")
    plt.ylabel("smoothed abs diff")
    plt.title("Smoothed auto-correlation at (152,152)")
    plt.grid(True)
    plt.savefig("figure/task4_smoothed_auto_correlation.png",
                bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 5: Cross-correlation
    # -----------------------------
    cc = cross_correlation(tb_left, tb_right)
    plt.plot(range(len(cc)), cc, marker="o")
    plt.xlabel("disparity d")
    plt.ylabel("smoothed abs diff")
    plt.title("Cross-correlation at (152,152)")
    plt.grid(True)
    plt.savefig("figure/task5_cross_correlation.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 6: Left -> Right disparity map
    # -----------------------------
    disp_L = disparity_map(tb_left, tb_right,
                           max_d=30, kernel_size=5, plot_result=False)
    disp_L_vis = (disp_L.astype(np.float32) / 30 * 255).astype(np.uint8)
    plt.imshow(disp_L_vis, cmap="gray")
    plt.title("Disparity L→R")
    plt.axis("off")
    plt.savefig("figure/task6_disparity_L2R.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 7: Right -> Left disparity map
    # -----------------------------
    disp_R = right_left_disparity(tb_left, tb_right,
                                  max_d=30, kernel_size=5, plot_result=False)
    disp_R_vis = (np.abs(disp_R).astype(np.float32) / 30 * 255).astype(np.uint8)
    plt.imshow(disp_R_vis, cmap="gray")
    plt.title("Disparity R→L (magnitude)")
    plt.axis("off")
    plt.savefig("figure/task7_disparity_R2L.png", bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 8: LR consistency check
    # -----------------------------
    clean = disparity_check(tb_left, tb_right,
                            max_d=30, kernel_size=5, plot_result=False)
    clean_vis = (clean.astype(np.float32) / 30 * 255).astype(np.uint8)
    plt.imshow(clean_vis, cmap="gray")
    plt.title("LR Consistency Cleaned Disparity")
    plt.axis("off")
    plt.savefig("figure/task8_lr_consistency_cleaned.png",
                bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Task 9: 3D reconstruction (depth vis)
    # -----------------------------
    # 建立 point cloud (kermit.ply)
    reconstruction(tb_left, tb_right, max_d=30, kernel_size=5,
                   ply_filename="kermit.ply")

    depth = clean.astype(np.float32)
    depth[depth == 0] = np.nan
    plt.imshow(depth, cmap="inferno")
    plt.title("Reconstruction depth visualization")
    plt.colorbar()
    plt.axis("off")
    plt.savefig("figure/task9_reconstruction_depth.png",
                bbox_inches="tight")
    plt.close()
