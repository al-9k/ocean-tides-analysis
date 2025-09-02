# utils/genfield.py 

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import correlate2d 

from utils.masks import * 
from config.params import *


def genfield(image1, image2, valid_mask, dt, meters_per_pixel,
             circle_mask, plot=True, image1_name=None, image2_name=None,
             save_path=None, return_grids=False, normalize=False):


    if normalize == True:
        image1 = (image1 - np.mean(image1)) / np.std(image1)
        image2 = (image2 - np.mean(image2)) / np.std(image2)


    h, w = image1.shape
    centers = [
        (y, x)
        for y in range(pad + ref_size // 2, h - pad - ref_size // 2, step)
        for x in range(pad + ref_size // 2, w - pad - ref_size // 2, step)
    ]
    ny = len(range(pad + ref_size // 2, h - pad - ref_size // 2, step))
    nx = len(range(pad + ref_size // 2, w - pad - ref_size // 2, step))

    u_grid = np.full((ny, nx), np.nan)
    v_grid = np.full((ny, nx), np.nan)
    coords = []
    angles = []
    velocities = []
    data = []
    zeroes = []


    if plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image1, cmap='gray', extent=[0, w * km_per_pixel, h * km_per_pixel, 0])
        ax.set_title(f'NCC Between images at {image1_name} and {image2_name}')
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.invert_yaxis()

    for idx, (y, x) in enumerate(centers):
        row = idx // nx
        col = idx % nx

        if not is_inside_radar_area(x, y, 450, 450, 450):
            continue

        if not is_valid_window_combined(x, y, ref_size, circle_mask, valid_mask, threshold=0.95):
            v_x = 0
            v_y = 0
            v_mag = 0
            u_grid[row, col] = v_mag
            v_grid[row, col] = v_mag
            coords.append((y, x))
            if plot:
                ax.scatter(x * km_per_pixel, y * km_per_pixel, color='red', s=10)
            continue

        ref_window = image1[y - ref_size // 2:y + ref_size // 2, x - ref_size // 2:x + ref_size // 2]
        search_window = image2[y - search_size // 2:y + search_size // 2, x - search_size // 2:x + search_size // 2]

        if ref_window.shape != (ref_size, ref_size) or search_window.shape != (search_size, search_size):
            continue

        std = np.std(ref_window)
        delta = ref_window.max() - ref_window.min()
        if std < 0.08 or delta < 1.0:
            v_x = 0
            v_y = 0
            v_mag = 0
            u_grid[row, col] = v_mag
            v_grid[row, col] = v_mag
            coords.append((y, x))
            if plot:
                ax.scatter(x * km_per_pixel, y * km_per_pixel, color='red', s=10)

            continue

        corr = correlate2d(search_window, ref_window, mode='valid')
        if corr.max() == corr.min():
            continue

        corr_norm = (corr - corr.min()) / (corr.max() - corr.min())
        peak_y, peak_x = np.unravel_index(np.argmax(corr_norm), corr_norm.shape)
        peak_value = corr_norm[peak_y, peak_x]

        neighborhood = ref_window[max(peak_y - 1, 0):peak_y + 2, max(peak_x - 1, 0):peak_x + 2]
        local_mean = (np.sum(neighborhood) - peak_value) / (neighborhood.size - 1)
        if (peak_value - local_mean) < prominence_threshold:
            v_x = 0
            v_y = 0
            v_mag = 0
            u_grid[row, col] = v_mag
            v_grid[row, col] = v_mag
            coords.append((y, x))
            if plot:
                ax.scatter(x * km_per_pixel, y * km_per_pixel, color='red', s=10)

            continue

        offset_y = peak_y - pad
        offset_x = peak_x - pad

        dx_m = offset_x * meters_per_pixel
        dy_m = offset_y * meters_per_pixel

        v_x = dx_m / dt
        v_y = dy_m / dt
        v_mag = np.sqrt(v_x**2 + v_y**2)
        angle_deg = np.degrees(np.arctan2(v_y, v_x))  # compute angle


        if not (min_v < v_mag < max_v):
            v_x = 0
            v_y = 0
            v_mag = 0
            u_grid[row, col] = v_mag
            v_grid[row, col] = v_mag
            zeroes.append((y, x))
            if plot:
                ax.scatter(x * km_per_pixel, y * km_per_pixel, color='red', s=10)

            continue

        u_grid[row, col] = v_x
        v_grid[row, col] = v_y
        coords.append((y, x))
        angles.append(angle_deg)
        velocities.append(v_mag)
        data.append((angle_deg, y, x, v_y, v_x))


        # if plot:
        #     if v_mag > 0:
        #       ax.arrow(x * km_per_pixel, y * km_per_pixel,
        #               v_x * km_per_pixel, v_y * km_per_pixel,
        #               head_width=0.25, head_length=0.35,
        #               fc='orange', ec='orange', linewidth=2)

    if len(angles) > 0:
        median_angle = np.median(angles)
        new_coords = []

        for i, (angle_deg, y, x, v_y, v_x) in enumerate(data):
            if np.abs(angle_deg - median_angle) < 45:
                if plot:
                    ax.arrow(x * km_per_pixel, y * km_per_pixel,
                            v_x * km_per_pixel, v_y * km_per_pixel,
                            head_width=0.25, head_length=0.35,
                            fc='orange', ec='orange', linewidth=2)
                new_coords.append(coords[i])  # Keep this coordinate
            # Else: skip adding the coordinate (effectively popping it)

        coords = new_coords  # Update coords to only valid ones

    if plot:
        plt.tight_layout()
        plt.show()
        if save_path:
            fig.savefig(save_path, dpi=300)

    if return_grids:
        return u_grid, v_grid
    else:
        return coords
