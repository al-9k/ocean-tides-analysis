# config/params.py

ref_size = 24
half_ref = ref_size // 2
pad = 4
search_size = ref_size + 2 * pad
meters_per_pixel = 25
km_per_pixel = meters_per_pixel / 1000
step = ref_size
max_v = 1
min_v = 0
dt = 240 * 15
prominence_threshold = 0.05