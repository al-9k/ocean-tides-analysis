# root/process_images.py

"""
process_images.py

Step 0 of the pipeline: Process raw radar images into daily NetCDF files.

What it does:
- Traverses input directories under /storage/fstdenis/RADAR/Barrow_RADAR/RAW_Data/YYYY/MM/DD/
- Each subfolder contains raw .tif images (4-minute exposures).
- Processes them in ~1-hour intervals (using pairs of images 1 hour apart).
- Normalizes images, computes velocity fields (u, v) via genfield().
- Applies masks (valid_mask, circle_mask) for domain filtering.
- Stacks results into time × y × x arrays.
- Writes daily NetCDFs under ./output/YYYY/MM/DD.nc

Key notes:
- Skips days that already have a saved NetCDF (avoids reprocessing).
- Handles shape mismatches or errors gracefully, skipping invalid pairs.
- Outputs CF-like structure with variables "u" and "v", and coordinates time, y, x.

Usage:
    python process_images.py
"""


import os
import numpy as np
import xarray as xr
from skimage import io
from tqdm import tqdm

from config.params import *
from utils.genfield import genfield
from utils.masks import *

input_root = '/storage/fstdenis/RADAR/Barrow_RADAR/RAW_Data' # Each image is a 4 minute exposure, but we will process them in hourly intervals
output_root = './output/'

def ensure_output_path(date_str):
    year = date_str[:4]
    month = date_str[4:6]
    output_dir = os.path.join(output_root, year, month)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'{date_str}.nc')

def main():
    # valid_mask, shape = load_mask('SIR.png')
    # circle_mask = generate_circle_mask(shape=shape, radius=450)

    for year in ['2022', '2023']:
        for month in sorted(os.listdir(os.path.join(input_root, year))):
            month_path = os.path.join(input_root, year, month)
            if not os.path.isdir(month_path):
                continue

            for day_folder in sorted(os.listdir(month_path)):
                day_path = os.path.join(month_path, day_folder)
                if not os.path.isdir(day_path):
                    continue

                date_str = day_folder
                out_path = ensure_output_path(date_str)
                if os.path.exists(out_path): # For debugging purposes
                    # If the file already exists, skip processing
                    # This is useful for debugging to avoid reprocessing
                    print(f"⏩ Skipping {date_str} — already processed.")
                    continue

                images = sorted([f for f in os.listdir(day_path) if f.endswith('.tif')])
                if len(images) < 2:
                    continue

                u_stack, v_stack = [], []

                for i in tqdm(range(0, len(images) - 15, 15), desc=f'Processing {date_str}'):
                    img1_path = os.path.join(day_path, images[i])
                    img2_path = os.path.join(day_path, images[i + 15])

                    img1 = io.imread(img1_path, as_gray=True)
                    img2 = io.imread(img2_path, as_gray=True)

                    img1 = (img1 - np.mean(img1)) / np.std(img1)
                    img2 = (img2 - np.mean(img2)) / np.std(img2)

                    # try: ==> commented out for debugging. Shape mismatch issues.
                    #     u, v = genfield(img1, img2, valid_mask, dt, meters_per_pixel,
                    #                     circle_mask, plot=False, return_grids=True)
                    #     u_stack.append(u)
                    #     v_stack.append(v) 

                    try: # hotfix for shape mismatch issues
                        # Ensure images are valid and have the same shape
                        if img1.shape != img2.shape:
                            print(f"⚠️ Skipping due to shape mismatch: {img1.shape} vs {img2.shape}")
                            continue

                        # Generate velocity fields
                        dt = 240 * 15  # Use the correct dt value from params.py
                        u, v = genfield(img1, img2, valid_mask, dt, meters_per_pixel,
                                        circle_mask, plot=False, return_grids=True)
                        
                        if not u_stack:
                            expected_shape = u.shape
                        elif u.shape != expected_shape or v.shape != expected_shape:
                            print(f"⚠️ Skipping due to mismatched shape: {u.shape} vs expected {expected_shape}")
                            continue

                        u_stack.append(u)
                        v_stack.append(v)

                    except Exception as e:
                        print(f"⚠️ Error processing {img1_path} and {img2_path}: {e}")
                        continue

                if not u_stack:
                    print(f"⚠️ No valid velocity data for {date_str}. Skipping.")
                    continue

                u_3d = np.stack(u_stack, axis=0)
                v_3d = np.stack(v_stack, axis=0)

                y_coords = np.arange(u_3d.shape[1]) * km_per_pixel
                x_coords = np.arange(u_3d.shape[2]) * km_per_pixel
                # t_coords = np.arange(u_3d.shape[0])  // 15 # Correcting for time steps. Hourly intervals.
                # t_coords = np.linspace(0, 23, len(u_stack))  # Assuming 24 steps for a day
                t_coords = np.arange(len(u_stack))  # Assuming each stack corresponds to a time step

                # # Ensure coordinates are 1D arrays
                # y_coords = np.asarray(y_coords).flatten()
                # x_coords = np.asarray(x_coords).flatten()
                # t_coords = np.asarray(t_coords).flatten()


                # Debugging output
                print(f"Shape of u_stack: {np.shape(u_stack)}")
                print(f"Shape of v_stack: {np.shape(v_stack)}")
                print(f"Shape of u_3d: {np.shape(u_3d)}")
                print(f"Shape of v_3d: {np.shape(v_3d)}")
                print(f"Shape of y_coords: {np.shape(y_coords)}")
                print(f"Shape of x_coords: {np.shape(x_coords)}")
                print(f"Shape of t_coords: {np.shape(t_coords)}")
                ds = xr.Dataset({
                    "u": (["time", "y", "x"], u_3d),
                    "v": (["time", "y", "x"], v_3d)
                }, coords={
                    "time": t_coords,
                    "y": y_coords,
                    "x": x_coords
                })

                out_path = ensure_output_path(date_str)
                ds.to_netcdf(out_path)
                print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
