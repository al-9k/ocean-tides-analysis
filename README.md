# Ocean Tides Analysis

This repository contains tools for processing and analyzing your data. Below is a step-by-step guide on how to use it.

---

## 1. Process Images

The first step is to run the `process_images.py` script.  
- Open the script and **update the input file path** as needed.  
- It is recommended to have **`.log`** file for printing and debugging.

```bash
python process_images.py
```

---

## 2. Merge Outputs

After processing is complete, run the `merger.py` script to merge the outputs from `process_images.py`.

```bash
python merger.py \
    --root output \
    --out  output/merged/time_series_uv.nc
```

- This will generate a merged file under `./output/merged/time_series_uv.nc` for subsequent analysis.

---

## 3. Analysis with Jupyter Notebook

Once the data is merged:  
- You can open the provided Jupyter notebook to explore and analyze the data.  
- A pre-merged file is already available under:

```
output/merged/time_series_uv.nc
```

- You can use this directly if you don’t need to re-run the merge step.

---

## Notes

- Make sure all dependencies (Python packages) are installed before running the scripts.  
- The workflow is sequential: `process_images.py` → `merge.py` → notebook analysis.
