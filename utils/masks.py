from skimage import io
import numpy as np

def load_mask(path):
    mask = io.imread(path)
    mask_channel = mask[:, :, 0]
    valid_mask = (mask_channel == 255)
    return valid_mask, mask_channel.shape

valid_mask, shape = load_mask(path = 'SIR.png')


def generate_circle_mask(shape, center=None, radius=450):
    h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    Y, X = np.ogrid[:h, :w]
    dist_from_center = (X - center[1])**2 + (Y - center[0])**2
    return dist_from_center <= radius**2
  
circle_mask = generate_circle_mask(shape, center=None, radius=450)

def is_valid_window_combined(x, y, window_size, circle_mask, valid_mask, threshold=0.9):
    half = window_size // 2

    y_start = y - half
    y_end = y + half + 1
    x_start = x - half
    x_end = x + half + 1

    if y_start < 0 or x_start < 0 or y_end > circle_mask.shape[0] or x_end > circle_mask.shape[1]:
        return False

    window_circle = circle_mask[y_start:y_end, x_start:x_end]
    window_valid = valid_mask[y_start:y_end, x_start:x_end]

    circle_ratio = np.mean(window_circle)
    valid_ratio = np.mean(window_valid)

    return circle_ratio >= threshold and valid_ratio >= threshold

def is_inside_radar_area(x, y, center_x, center_y, radar_radius):
    return (x - center_x)**2 + (y - center_y)**2 <= radar_radius**2
