"""CS180 Project 1: Prokudin--Gorskii reconstruction.

The implementation is intentionally self contained: channels are split from the
vertical B/G/R plate, a Gaussian-like 2x2 average pyramid is built, and a
coarse-to-fine translation search is performed with normalized cross-correlation
on both intensity and edge magnitude.  The same fixed parameters are used for
all files.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if _HERE.name == "code" else _HERE
_DATA_ROOT = _ROOT / "CS180_fa2026_proj1_data"
# Accept both the original zip's nested folder and the flattened course folder.
DATA = (_DATA_ROOT / "CS180_fa2026_proj1_data" if (_DATA_ROOT / "CS180_fa2026_proj1_data").exists() else _DATA_ROOT)
OUT = _ROOT / "results"

# The three optional examples in this report are downloaded directly from the
# Library of Congress.  Keeping the provenance next to the implementation
# makes the report auditable while still allowing the code to process any
# compatible TIFF supplied with --input-dir.
LOC_METADATA = {
    "loc_2018679023": {
        "collection_item": "2018679023",
        "source_url": "https://www.loc.gov/item/2018679023/",
        "plate_url": "https://tile.loc.gov/storage-services/master/pnp/prok/00100/00159u.tif",
        "title": "V Iasnoe polianie",
    },
    "loc_2018679024": {
        "collection_item": "2018679024",
        "source_url": "https://www.loc.gov/item/2018679024/",
        "plate_url": "https://tile.loc.gov/storage-services/master/pnp/prok/00100/00160u.tif",
        "title": "V Iasnoe polianie (variant)",
    },
    "loc_2018679025": {
        "collection_item": "2018679025",
        "source_url": "https://www.loc.gov/item/2018679025/",
        "plate_url": "https://tile.loc.gov/storage-services/master/pnp/prok/00100/00161u.tif",
        "title": "Na Uralie",
    },
}


def load_plate(path: Path) -> np.ndarray:
    """Read a plate as float32 in [0, 1], splitting vertical B,G,R thirds."""
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim != 2:
        a = a[..., 0]
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    # Ignore a possible one-pixel remainder instead of silently mixing plates.
    h = (a.shape[0] // 3) * 3
    a = a[:h]
    return np.stack(np.split(a, 3, axis=0), axis=0)  # B, G, R


def down2(a: np.ndarray) -> np.ndarray:
    """Anti-aliased 2x downsampling, written without a high-level pyramid call."""
    h, w = a.shape[-2:]
    h2, w2 = h // 2, w // 2
    a = a[..., : 2 * h2, : 2 * w2]
    return (a[..., 0::2, 0::2] + a[..., 1::2, 0::2] +
            a[..., 0::2, 1::2] + a[..., 1::2, 1::2]) * 0.25


def pyramid(a: np.ndarray, levels: int = 6) -> list[np.ndarray]:
    out = [a]
    while len(out) < levels and min(out[-1].shape) >= 160:
        out.append(down2(out[-1]))
    return out[::-1]  # coarsest first


def edge_mag(a: np.ndarray) -> np.ndarray:
    """Fast Sobel-like magnitude using central differences."""
    gy = np.zeros_like(a)
    gx = np.zeros_like(a)
    gy[1:-1] = a[2:] - a[:-2]
    gx[:, 1:-1] = a[:, 2:] - a[:, :-2]
    return np.sqrt(gx * gx + gy * gy + 1e-8)


def ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    x = x - float(x.mean())
    y = y - float(y.mean())
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-8)
    return float(np.sum(x * y) / den)


def l2_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Brightness-normalized L2/RMSE, lower is better.

    Each plate is z-normalized over the same overlap before measuring the
    Euclidean distance.  This follows the course's L2 option while avoiding a
    meaningless penalty from different historical filter exposures.
    """
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    x = (x - float(x.mean())) / (float(x.std()) + 1e-8)
    y = (y - float(y.mean())) / (float(y.std()) + 1e-8)
    return float(np.sqrt(np.mean((x - y) ** 2)))


def pair_metrics(base: np.ndarray, moving: np.ndarray, dy: int, dx: int) -> tuple[float, float]:
    """NCC and normalized L2 on the exact non-wrapping overlap."""
    h, w = base.shape
    margin = max(abs(dy), abs(dx)) + 8
    y0, y1 = max(margin, dy + margin), min(h - margin, h + dy - margin)
    x0, x1 = max(margin, dx + margin), min(w - margin, w + dx - margin)
    if y1 <= y0 or x1 <= x0:
        return -1.0, float("inf")
    b = base[y0:y1:2, x0:x1:2]
    m = moving[y0 - dy:y1 - dy:2, x0 - dx:x1 - dx:2]
    return ncc(b, m), l2_rmse(b, m)


def score_shift(base: np.ndarray, moving: np.ndarray, dy: int, dx: int,
                stride: int = 4) -> float:
    """NCC score on a non-wrapping overlap; 70% intensity, 30% edges."""
    h, w = base.shape
    margin = max(abs(dy), abs(dx)) + 8
    y0, y1 = max(margin, dy + margin), min(h - margin, h + dy - margin)
    x0, x1 = max(margin, dx + margin), min(w - margin, w + dx - margin)
    if y1 <= y0 or x1 <= x0:
        return -1.0
    b = base[y0:y1:stride, x0:x1:stride]
    m = moving[y0 - dy:y1 - dy:stride, x0 - dx:x1 - dx:stride]
    if b.size < 100:
        return -1.0
    s_int = ncc(b, m)
    eb, em = edge_mag(b), edge_mag(m)
    s_edge = ncc(eb, em)
    return 0.70 * s_int + 0.30 * s_edge


def align_single(base: np.ndarray, moving: np.ndarray, radius: int = 15) -> tuple[int, int, float]:
    best = (-10.0, 0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            s = score_shift(base, moving, dy, dx, stride=2)
            if s > best[0]:
                best = (s, dy, dx)
    return best[1], best[2], best[0]


def align_pyramid(base: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    bp, mp = pyramid(base), pyramid(moving)
    dy = dx = 0
    score = 0.0
    for level, (b, m) in enumerate(zip(bp, mp)):
        if level:
            dy, dx = 2 * dy, 2 * dx
        # A slightly wider coarse window prevents the low-resolution aliasing
        # failure seen on the strongly color-dependent Emir plate.
        radius = 24 if level == 0 else (7 if level == 1 else 5)
        # At fine levels score around the upsampled estimate.
        best = (-10.0, dy, dx)
        for ddy in range(-radius, radius + 1):
            for ddx in range(-radius, radius + 1):
                cy, cx = dy + ddy, dx + ddx
                s = score_shift(b, m, cy, cx, stride=4 if level < len(bp) - 1 else 3)
                if s > best[0]:
                    best = (s, cy, cx)
        score, dy, dx = best
    return int(dy), int(dx), float(score)


def phase_shift(base: np.ndarray, moving: np.ndarray, factor: int = 4) -> tuple[int, int, float]:
    """FFT phase correlation on edge maps, robust to per-filter brightness.

    The cross-power spectrum discards amplitude and keeps only the translation
    phase.  A Hann window suppresses the plate boundary, and the returned peak
    is converted from the downsampled grid to full-resolution pixels.
    """
    a = edge_mag(base[::factor, ::factor]).astype(np.float32)
    b = edge_mag(moving[::factor, ::factor]).astype(np.float32)
    h, w = a.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fa * np.conj(fb)
    corr = np.fft.ifft2(cross / (np.abs(cross) + 1e-8)).real
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = int(py if py <= h // 2 else py - h) * factor
    dx = int(px if px <= w // 2 else px - w) * factor
    return dy, dx, float(corr[py, px])


def align_registration(base: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    """Phase-correlation initialization followed by a local NCC/edge refine."""
    factor = 1 if min(base.shape) < 1000 else 4
    py, px, _ = phase_shift(base, moving, factor)
    # Phase correlation is global; a small full-resolution search removes its
    # integer-grid quantization and rejects a spurious edge peak.
    radius = 8 if factor == 1 else 6
    best = (-10.0, py, px)
    for dy in range(py - radius, py + radius + 1):
        for dx in range(px - radius, px + radius + 1):
            s = score_shift(base, moving, dy, dx, stride=2 if factor == 1 else 6)
            if s > best[0]:
                best = (s, dy, dx)
    return int(best[1]), int(best[2]), float(best[0])


def shift_no_wrap(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate with black fill (never circularly wraps plate edges)."""
    h, w = a.shape
    out = np.zeros_like(a)
    if dy >= 0:
        sy0, sy1, ty0, ty1 = 0, h - dy, dy, h
    else:
        sy0, sy1, ty0, ty1 = -dy, h, 0, h + dy
    if dx >= 0:
        sx0, sx1, tx0, tx1 = 0, w - dx, dx, w
    else:
        sx0, sx1, tx0, tx1 = -dx, w, 0, w + dx
    if sy1 > sy0 and sx1 > sx0 and ty1 > ty0 and tx1 > tx0:
        out[ty0:ty1, tx0:tx1] = a[sy0:sy1, sx0:sx1]
    return out


def valid_box(shifts: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    # Intersection where every translated channel has real source samples.
    top = max(0, max(dy for dy, _ in shifts))
    bottom = min(h, min(h + dy for dy, _ in shifts))
    left = max(0, max(dx for _, dx in shifts))
    right = min(w, min(w + dx for _, dx in shifts))
    return top, bottom, left, right


def automatic_crop(rgb: np.ndarray, shifts: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop overlap plus detected plate margins using a central-scene baseline."""
    h, w = rgb.shape[:2]
    t, b, l, r = valid_box(shifts, (h, w))
    view = rgb[t:b, l:r]
    gray = view.mean(axis=2)
    disagreement = view.std(axis=2)
    H, W = gray.shape
    ys, ye = max(2, H // 5), min(H - 2, 4 * H // 5)
    xs, xe = max(2, W // 5), min(W - 2, 4 * W // 5)
    central_gray = gray[ys:ye, xs:xe]
    central_d = disagreement[ys:ye, xs:xe]
    d_ref = max(0.025, float(np.median(central_d)))
    extreme_ref = max(0.025, float(((central_gray < 0.018) | (central_gray > 0.982)).mean()))
    d_limit = max(0.105, d_ref * 1.85)
    extreme_limit = max(0.10, extreme_ref * 1.8)
    def bad_row(row):
        extreme = float(((row < 0.018) | (row > 0.982)).mean())
        mean, spread = float(row.mean()), float(row.std())
        low_information = spread < 0.12 and (mean < 0.12 or mean > 0.92)
        return (spread < 0.035 and (mean < 0.035 or mean > 0.965)) or low_information or extreme > extreme_limit
    def bad_col(col, dcol):
        extreme = float(((col < 0.018) | (col > 0.982)).mean())
        mean, spread = float(col.mean()), float(col.std())
        uniform_extreme = spread < 0.035 and (mean < 0.035 or mean > 0.965)
        low_information = spread < 0.12 and (mean < 0.12 or mean > 0.92)
        return float(dcol.mean()) > d_limit or uniform_extreme or low_information or extreme > extreme_limit
    def first_good(flags, cap, run=12):
        for i in range(min(cap, len(flags))):
            if not any(flags[i:min(len(flags), i + run)]): return i
        return min(cap, len(flags))
    cap_y, cap_x = max(8, H // 8), max(8, W // 8)
    row_flags = [bad_row(gray[i]) or float(disagreement[i].mean()) > d_limit for i in range(H)]
    top = first_good(row_flags, cap_y)
    bottom = H - first_good(row_flags[::-1], cap_y)
    col_flags = [bad_col(gray[:, i], disagreement[:, i]) for i in range(W)]
    left = first_good(col_flags, cap_x)
    right = W - first_good(col_flags[::-1], cap_x)
    t2, b2, l2, r2 = t + top, t + bottom, l + left, l + right
    # A narrow scan frame can contain a colored registration strip immediately
    # before an otherwise white run.  A small data-independent safety guard
    # removes that final sliver without materially changing the photograph.
    guard_y, guard_x = max(4, int(0.010 * (b2 - t2))), max(8, int(0.025 * (r2 - l2)))
    t2, b2 = min(b2 - 1, t2 + guard_y), max(t2 + 1, b2 - guard_y)
    l2, r2 = min(r2 - 1, l2 + guard_x), max(l2 + 1, r2 - guard_x)
    return rgb[t2:b2, l2:r2], (int(t2), int(b2), int(l2), int(r2))


def color_correct(rgb: np.ndarray) -> np.ndarray:
    """Percentile contrast + restrained gray-world white balance + gamma."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [1.0, 99.0])
        out[..., c] = np.clip((rgb[..., c] - lo) / max(hi - lo, 1e-6), 0, 1)
    means = np.mean(out.reshape(-1, 3), axis=0)
    target = float(np.mean(means))
    gains = np.clip(target / np.maximum(means, 1e-5), 0.85, 1.18)
    out *= gains[None, None, :]
    out = np.clip(out, 0, 1)
    return np.power(out, 0.96)


def save_jpeg(a: np.ndarray, path: Path, quality: int = 93, max_side: int = 1800) -> None:
    h, w = a.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        im = Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255))
    im.save(path, quality=quality, optimize=True)


def process(path: Path, out_dir: Path) -> dict:
    plates = load_plate(path)
    base = plates[0]
    # G and R are aligned to B; displacement is the shift applied to that plate.
    gy, gx, gs = align_registration(base, plates[1])
    ry, rx, rs = align_registration(base, plates[2])
    shifts = [(0, 0), (gy, gx), (ry, rx)]
    shifted = np.stack([shift_no_wrap(p, *s) for p, s in zip(plates, shifts)], axis=-1)
    cropped, box = automatic_crop(shifted, shifts)
    corrected = color_correct(cropped)
    stem = path.stem
    save_jpeg(shifted, out_dir / f"{stem}_raw.jpg")
    save_jpeg(corrected, out_dir / f"{stem}.jpg")
    single = None
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        sgy, sgx, sgs = align_single(base, plates[1], radius=15)
        sry, srx, srs = align_single(base, plates[2], radius=15)
        single_shifts = [(0, 0), (sgy, sgx), (sry, srx)]
        single_rgb = np.stack([shift_no_wrap(p, *s) for p, s in zip(plates, single_shifts)], axis=-1)
        save_jpeg(single_rgb, out_dir / f"{stem}_single.jpg")
        sgn, sgl2 = pair_metrics(base, plates[1], sgy, sgx)
        srn, srl2 = pair_metrics(base, plates[2], sry, srx)
        single = {"G_to_B": [int(sgy), int(sgx)], "R_to_B": [int(sry), int(srx)],
                  "ncc": {"G": round(sgn, 5), "R": round(srn, 5)},
                  "l2_rmse": {"G": round(sgl2, 5), "R": round(srl2, 5)},
                  "output": f"{stem}_single.jpg"}
    # A compact before/after comparison for the report.
    save_jpeg(cropped, out_dir / f"{stem}_before.jpg", max_side=1200)
    gn, gl2 = pair_metrics(base, plates[1], gy, gx)
    rn, rl2 = pair_metrics(base, plates[2], ry, rx)
    record = {
        "name": stem, "source": path.name, "height": int(plates.shape[1]), "width": int(plates.shape[2]),
        "offsets": {"G_to_B": [int(gy), int(gx)], "R_to_B": [int(ry), int(rx)]},
        "ncc": {"G": round(gn, 5), "R": round(rn, 5)},
        "l2_rmse": {"G": round(gl2, 5), "R": round(rl2, 5)},
        "summary_metrics": {"mean_ncc": round((gn + rn) / 2.0, 5),
                             "mean_l2_rmse": round((gl2 + rl2) / 2.0, 5)},
        "crop_box": list(box), "output": f"{stem}.jpg", "before": f"{stem}_before.jpg",
        "single_scale": single,
        "data_group": "loc_collection" if stem in LOC_METADATA else "course_samples",
    }
    if stem in LOC_METADATA:
        record.update(LOC_METADATA[stem])
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="optional filenames")
    ap.add_argument("--input-dir", type=Path, default=DATA,
                    help="directory containing vertical B/G/R plates")
    ap.add_argument("--output-dir", type=Path, default=OUT,
                    help="directory for reconstructed JPEGs and results.json")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or [p.name for p in sorted(args.input_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"}]
    records = []
    for name in names:
        p = args.input_dir / name
        print(f"processing {name} ...", flush=True)
        records.append(process(p, args.output_dir))
        print(records[-1]["offsets"], "crop", records[-1]["crop_box"], flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
