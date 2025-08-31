#!/usr/bin/env python3
"""
merge_uv_timeseries.py

Step 1 of the pipeline: Merge all daily NetCDF files into a single dataset
with a correct, CF-compliant time coordinate that spans 2022-01-01 to 2023-05-31
(based on the files that exist). Writes: ./output/merged/time_series_uv.nc

What it does:
- Globs daily files under ./output/YYYY/MM/*.nc
- Opens each file. If its "time" coordinate is already decoded to datetime64, uses it.
  Otherwise, reconstructs time from the filename date + hourly offsets (0..n-1).
- Sorts & de-duplicates time within each file.
- Concatenates over "time" and writes a CF-compliant NetCDF with proper time encoding.
- Prints summary diagnostics (start/end, length, gaps).

Usage example:
    python merger.py \
        --root output \
        --out  output/merged/time_series_uv.nc
"""
import argparse
import glob
import os
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import xarray as xr


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.nc$")


def extract_date_from_path(path: str) -> pd.Timestamp:
    m = DATE_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract YYYY-MM-DD from filename: {path}")
    # Treat as UTC-naive (no timezone); adjust if your data is local time.
    return pd.to_datetime(m.group(1))


def is_datetime64(da: xr.DataArray) -> bool:
    return np.issubdtype(da.dtype, np.datetime64)


def open_one_file(path: str) -> xr.Dataset:
    """
    Open a single daily file. If "time" is not CF-decoded to datetime64,
    rebuild it from the filename date + hourly offsets (0..n-1).
    """
    # First try: rely on CF decoding
    ds = xr.open_dataset(path, decode_times=True)
    needs_rebuild = ("time" not in ds.coords) or (not is_datetime64(ds["time"]))

    if needs_rebuild:
        # Fallback: reconstruct from filename date + hourly offsets
        ds_raw = xr.open_dataset(path, decode_times=False)

        if "time" in ds_raw.dims:
            n = ds_raw.sizes["time"]
        else:
            # Infer time length from u or v if time is not declared as a dimension (unlikely)
            if "u" in ds_raw:
                n = ds_raw["u"].shape[0]
            elif "v" in ds_raw:
                n = ds_raw["v"].shape[0]
            else:
                raise ValueError(f"No time dimension and neither 'u' nor 'v' present in {path}")

        day0 = extract_date_from_path(path)
        # Assume 1-hour interval; if your native cadence differs, change freq here.
        times = day0 + pd.to_timedelta(np.arange(n), unit="h")

        # Assign rebuilt time (ensure it's strictly increasing and unique)
        ds = ds_raw.assign_coords(time=pd.DatetimeIndex(times))
        ds = ds.sortby("time")
        # Deduplicate within the file if needed
        _, unique_idx = np.unique(ds["time"].values, return_index=True)
        if len(unique_idx) != ds.sizes["time"]:
            ds = ds.isel(time=np.sort(unique_idx))
    else:
        # Ensure sorted & unique within-file times
        ds = ds.sortby("time")
        _, unique_idx = np.unique(ds["time"].values, return_index=True)
        if len(unique_idx) != ds.sizes["time"]:
            ds = ds.isel(time=np.sort(unique_idx))

    # Keep only expected variables + coords to avoid concat conflicts
    keep_vars = [v for v in ["u", "v"] if v in ds.variables]
    keep_coords = [c for c in ["time", "y", "x"] if c in ds.coords]
    ds = ds[keep_vars + keep_coords]

    return ds


def find_files(root: str) -> list[str]:
    # Expect files under root/YYYY/MM/*.nc
    pattern = os.path.join(root, "*", "*", "*.nc")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No NetCDF files found under {pattern}")
    # Sort by date parsed from filename to be safe
    files_sorted = sorted(files, key=lambda p: extract_date_from_path(p))
    return files_sorted


def concat_all(files: list[str]) -> xr.Dataset:
    # Open one by one to allow per-file fixes, then concat on time
    datasets = []
    for i, f in enumerate(files, 1):
        ds = open_one_file(f)
        # Strict dimension checks
        for dim in ["y", "x"]:
            if dim not in ds.dims:
                raise ValueError(f"Missing dimension '{dim}' in {f}")
        datasets.append(ds)

    ds_all = xr.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="outer",
    )

    # Final sort/unique on time across all files
    ds_all = ds_all.sortby("time")
    tvals = pd.DatetimeIndex(ds_all["time"].values)
    mask = ~tvals.duplicated()
    if (~mask).any():
        ds_all = ds_all.isel(time=mask)

    return ds_all


def print_summary(ds: xr.Dataset) -> None:
    t = pd.DatetimeIndex(ds["time"].values)
    print("----- MERGE SUMMARY -----")
    print(f"Total timesteps: {len(t)}")
    if len(t):
        print(f"Start: {t[0].isoformat()}")
        print(f"End:   {t[-1].isoformat()}")
        # Simple gap finder (expects hourly cadence)
        deltas = t[1:] - t[:-1]
        gaps = np.where(deltas > pd.Timedelta(hours=1))[0]
        if len(gaps):
            print(f"Detected {len(gaps)} gaps > 1h. First few indices: {gaps[:10].tolist()}")
        else:
            print("No gaps > 1h detected.")
    print("-------------------------")


def main():
    ap = argparse.ArgumentParser(description="Merge daily NetCDFs and fix/normalize time.")
    ap.add_argument("--root", default="/aos/home/ashnoot/radar/netcdf_output",
                    help="Root folder containing YYYY/MM/*.nc")
    ap.add_argument("--out", default="/aos/home/ashnoot/radar/netcdf_output/time_series_uv.nc",
                    help="Output NetCDF path")
    ap.add_argument("--compress", action="store_true",
                    help="Enable zlib compression for u,v")
    args = ap.parse_args()

    files = find_files(args.root)
    ds = concat_all(files)

    # Ensure standard CF encoding for time when writing
    encoding = {}
    if "u" in ds:
        encoding["u"] = {"zlib": args.compress, "complevel": 4} if args.compress else {}
    if "v" in ds:
        encoding["v"] = {"zlib": args.compress, "complevel": 4} if args.compress else {}
    # Be explicit about time encoding to avoid 1970-01-01 raw integers problem
    encoding["time"] = {
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "standard",
    }

    # Basic attrs (optional)
    ds = ds.assign_attrs({
        "title": "Merged hourly u,v timeseries",
        "institution": "Your Lab",
        "history": f"Merged on {datetime.utcnow().isoformat()}Z",
    })
    for v in ["u", "v"]:
        if v in ds:
            ds[v].attrs.setdefault("long_name", v)
            ds[v].attrs.setdefault("units", "unknown")

    # Write to NetCDF
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ds.to_netcdf(args.out, encoding=encoding, mode="w")

    # Print summary
    print_summary(ds)
    print(f"Written: {args.out}")


if __name__ == "__main__":
    main()

