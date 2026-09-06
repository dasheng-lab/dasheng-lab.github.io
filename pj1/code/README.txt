CS180 Project 1 - Reconstructing Prokudin-Gorskii Plates

Run from the project directory:

    python code/main.py

The script reads the course data from CS180_fa2026_proj1_data/ and writes
results to results/. It accepts both the nested folder produced by the course
zip and a flattened data folder. To process another collection of plates use,
for example:

    python code/main.py --input-dir loc_extra --output-dir results/loc

The B/G/R thirds are aligned to the blue plate. The full-resolution method
uses FFT phase correlation on edge maps for a robust global translation
estimate, followed by a small NCC + edge local refinement. JPEG files
additionally produce the required exhaustive single-scale NCC baseline.

The post-processing stages are deliberately separate and fixed for every
image: overlap-aware automatic cropping, a central-scene border detector that
requires a consecutive run of photographic strips, 1st/99th percentile
contrast, restrained gray-world white balance, and mild gamma correction. The
algorithm does not contain image-specific offsets or hand-edited crops.

Outputs include <stem>.jpg (final), <stem>_before.jpg (aligned before color
cleanup), <stem>_raw.jpg (before crop), and results.json (offsets, NCC,
brightness-normalized L2/RMSE, provenance, and crop boxes). The report is in
web/index.html; the Gradescope web artifact is web/page.pdf. The report also
contains three additional raw TIFF examples linked to their Library of
Congress Prokudin-Gorskii item pages and exact plate URLs.
