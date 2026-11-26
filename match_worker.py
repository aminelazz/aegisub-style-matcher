#!/usr/bin/env python3
"""
match_worker.py — Local CPU video frame sampling & ASS style extraction helper
for the Aegisub Style Matcher Automation 4 macro (style_matcher.lua).

Samples a 2-point probe ray from Point 1 (core text fill) across the text
border into Point 2 (surrounding background plate) to extract:
  - Primary Fill Color (\1c)
  - Border / Outline Color (\3c)
  - Edge Blur (\blur) via Gaussian/Sigmoid CDF slope fitting
  - Film Grain Opacity & Optional Vector Overlay (\alpha / \p1)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _str_to_bool(val: str) -> bool:
    """Parse flexible CLI boolean representations."""
    if isinstance(val, bool):
        return val
    v = str(val).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {val!r}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASS fill, border, blur, and film grain from a video frame."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--frame", type=int, required=True, help="0-based frame index.")
    parser.add_argument("--playresx", type=float, default=0.0, help="Script PlayResX.")
    parser.add_argument("--playresy", type=float, default=0.0, help="Script PlayResY.")
    parser.add_argument("--x1", type=float, required=True, help="Point 1 X (core fill).")
    parser.add_argument("--y1", type=float, required=True, help="Point 1 Y (core fill).")
    parser.add_argument("--x2", type=float, required=True, help="Point 2 X (background).")
    parser.add_argument("--y2", type=float, required=True, help="Point 2 Y (background).")
    parser.add_argument(
        "--radius", type=int, default=2, help="Sample patch radius in pixels (1..5)."
    )
    parser.add_argument(
        "--match-primary",
        type=_str_to_bool,
        default=True,
        help="Whether to extract \\1c primary color.",
    )
    parser.add_argument(
        "--match-border",
        type=_str_to_bool,
        default=True,
        help="Whether to extract \\3c border color.",
    )
    parser.add_argument(
        "--match-blur",
        type=_str_to_bool,
        default=True,
        help="Whether to estimate \\blur edge softness.",
    )
    parser.add_argument(
        "--match-grain",
        type=_str_to_bool,
        default=True,
        help="Whether to analyze background film grain.",
    )
    return parser.parse_args(argv)


def load_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Open video via OpenCV and seek/read the target BGR frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_idx = max(0, int(frame_idx))
        if total_frames > 0 and target_idx >= total_frames:
            target_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback: try last valid frame if seek overshot
            if total_frames > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 2))
                ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from video: {video_path}"
            )
        return frame
    finally:
        cap.release()


def scale_coords(
    x: float,
    y: float,
    playresx: float,
    playresy: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[float, float]:
    """Scale ASS script coordinates to native video pixel coordinates."""
    sx = (frame_w / playresx) if playresx > 0 else 1.0
    sy = (frame_h / playresy) if playresy > 0 else 1.0
    px = float(np.clip(x * sx, 0.0, max(0, frame_w - 1)))
    py = float(np.clip(y * sy, 0.0, max(0, frame_h - 1)))
    return px, py


def sample_patch_bgr(
    frame: np.ndarray, cx: float, cy: float, radius: int
) -> np.ndarray:
    """Average BGR values in a (2*radius+1)x(2*radius+1) patch around (cx, cy)."""
    h, w = frame.shape[:2]
    ix = int(round(cx))
    iy = int(round(cy))
    x0 = max(0, ix - radius)
    x1 = min(w, ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(h, iy + radius + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        ix_c = int(np.clip(ix, 0, w - 1))
        iy_c = int(np.clip(iy, 0, h - 1))
        return frame[iy_c, ix_c].astype(np.float64)
    return patch.reshape(-1, 3).mean(axis=0).astype(np.float64)


def bgr_to_ass_hex(bgr: np.ndarray) -> str:
    """Convert BGR triplet [0..255] to ASS hex format &HBBGGRR&."""
    b, g, r = [int(np.clip(round(float(c)), 0, 255)) for c in bgr[:3]]
    return f"&H{b:02X}{g:02X}{r:02X}&"


def bgr_to_luma(bgr: np.ndarray) -> np.ndarray:
    """Compute BT.709 luminance from BGR array (..., 3)."""
    return 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]


def _bilinear_sample_bgr(frame: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinearly sample BGR color at floating-point (x, y)."""
    h, w = frame.shape[:2]
    x = float(np.clip(x, 0.0, w - 1.0))
    y = float(np.clip(y, 0.0, h - 1.0))
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(w - 1, x0 + 1)
    y1 = min(h - 1, y0 + 1)
    fx = x - x0
    fy = y - y0

    c00 = frame[y0, x0].astype(np.float64)
    c10 = frame[y0, x1].astype(np.float64)
    c01 = frame[y1, x0].astype(np.float64)
    c11 = frame[y1, x1].astype(np.float64)

    return (
        c00 * (1.0 - fx) * (1.0 - fy)
        + c10 * fx * (1.0 - fy)
        + c01 * (1.0 - fx) * fy
        + c11 * fx * fy
    )


def sample_ray_profile(
    frame: np.ndarray,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    perp_radius: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a dense 1D BGR profile along the ray from p1 to p2.

    Averages across a thin perpendicular strip (+-perp_radius pixels) so
    random grain is attenuated without smearing the edge along the ray axis.

    Returns:
        dists: 1D array of distances in pixels from p1, shape (N,)
        colors: 2D array of BGR colors along the ray, shape (N, 3)
        luma: 1D array of BT.709 luminance along the ray, shape (N,)
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        c = sample_patch_bgr(frame, x1, y1, perp_radius)
        dists = np.array([0.0], dtype=np.float64)
        colors = c[None, :]
        return dists, colors, bgr_to_luma(colors)

    n_samples = max(int(math.ceil(length * 2.0)) + 1, 24)
    ts = np.linspace(0.0, 1.0, n_samples)
    dists = ts * length

    # Unit normal perpendicular to the ray
    nx = -dy / length
    ny = dx / length
    offsets = np.arange(-perp_radius, perp_radius + 1, dtype=np.float64)

    colors = np.zeros((n_samples, 3), dtype=np.float64)
    for i, t in enumerate(ts):
        cx = x1 + t * dx
        cy = y1 + t * dy
        acc = np.zeros(3, dtype=np.float64)
        for off in offsets:
            acc += _bilinear_sample_bgr(frame, cx + off * nx, cy + off * ny)
        colors[i] = acc / len(offsets)

    luma = bgr_to_luma(colors)
    return dists, colors, luma


def detect_border_and_edge(
    dists: np.ndarray,
    colors: np.ndarray,
    fill_bgr: np.ndarray,
    bg_bgr: np.ndarray,
) -> Tuple[np.ndarray, bool, int, int]:
    """
    Analyze the 1D color gradient vector from P1 (fill) to P2 (background) to
    locate the outline stroke (\3c) and the primary transition window for blur.

    Returns:
        border_bgr: Detected border BGR color.
        has_distinct_border: True if a separate outline color plateau was detected.
        edge_start_idx: Start index in ray for edge-blur fitting.
        edge_end_idx: End index in ray for edge-blur fitting.
    """
    n = len(dists)
    if n < 5:
        return bg_bgr.copy(), False, 0, n - 1

    # Central-difference color gradient magnitude along the ray (per sample step)
    grad = np.zeros(n, dtype=np.float64)
    grad[1:-1] = np.linalg.norm(colors[2:] - colors[:-2], axis=1) * 0.5
    grad[0] = np.linalg.norm(colors[1] - colors[0])
    grad[-1] = np.linalg.norm(colors[-1] - colors[-2])

    # Distance of each ray sample to the fill color and background baseline
    d_fill = np.linalg.norm(colors - fill_bgr[None, :], axis=1)
    d_bg = np.linalg.norm(colors - bg_bgr[None, :], axis=1)
    min_endpoint_dist = np.minimum(d_fill, d_bg)

    # Also compute orthogonal distance from the line segment (fill_bgr -> bg_bgr)
    seg = bg_bgr - fill_bgr
    seg_norm = float(np.linalg.norm(seg))
    if seg_norm > 1e-3:
        u_seg = seg / seg_norm
        rel = colors - fill_bgr[None, :]
        proj = np.clip((rel @ u_seg)[:, None], 0.0, seg_norm) * u_seg[None, :]
        orth_dist = np.linalg.norm(rel - proj, axis=1)
    else:
        orth_dist = min_endpoint_dist.copy()

    # Search interior region (avoid endpoints right inside fill or background)
    i_lo = max(1, int(round(0.08 * n)))
    i_hi = min(n - 1, int(round(0.92 * n)))
    if i_hi <= i_lo:
        i_lo, i_hi = 1, n - 1

    # Find local maxima of gradient
    peak_indices = [
        i
        for i in range(i_lo, i_hi)
        if grad[i] >= grad[i - 1] and grad[i] >= grad[i + 1] and grad[i] > 2.0
    ]

    # Check if there is a distinct border stroke between fill and background:
    # Either (a) significant orthogonal deviation from fill->bg line, or
    # (b) two well-separated gradient peaks with an interior plateau away from both endpoints.
    max_orth_idx = i_lo + int(np.argmax(orth_dist[i_lo:i_hi]))
    max_min_idx = i_lo + int(np.argmax(min_endpoint_dist[i_lo:i_hi]))

    has_distinct_border = False
    border_center_idx = max_min_idx

    if orth_dist[max_orth_idx] >= 14.0:
        has_distinct_border = True
        # Prefer the point near max orthogonal deviation where gradient is minimal (plateau center)
        window = max(2, int(round(0.08 * n)))
        w0 = max(i_lo, max_orth_idx - window)
        w1 = min(i_hi, max_orth_idx + window + 1)
        score = orth_dist[w0:w1] - 0.5 * grad[w0:w1]
        border_center_idx = w0 + int(np.argmax(score))
    elif len(peak_indices) >= 2:
        # Sort gradient peaks descending and inspect top 2 separated peaks
        sorted_peaks = sorted(peak_indices, key=lambda idx: grad[idx], reverse=True)
        p_a = sorted_peaks[0]
        p_b = None
        min_sep = max(3, int(round(0.08 * n)))
        for cand in sorted_peaks[1:]:
            if abs(cand - p_a) >= min_sep:
                p_b = cand
                break
        if p_b is not None:
            left_p, right_p = min(p_a, p_b), max(p_a, p_b)
            mid_slice = min_endpoint_dist[left_p : right_p + 1]
            if mid_slice.size > 0 and float(np.max(mid_slice)) >= 14.0:
                has_distinct_border = True
                # Pick interior point maximizing min_endpoint_dist while avoiding peaks
                mid_grad = grad[left_p : right_p + 1]
                score = mid_slice - 0.5 * mid_grad
                border_center_idx = left_p + int(np.argmax(score))

    if has_distinct_border:
        # Average a tight neighborhood around the border plateau center
        half_w = max(1, min(3, int(round(0.03 * n))))
        b0 = max(0, border_center_idx - half_w)
        b1 = min(n, border_center_idx + half_w + 1)
        border_bgr = colors[b0:b1].mean(axis=0)

        # Pick the primary edge window: from fill (0) up to the border plateau center
        # (or the border->bg edge if that has a cleaner/larger step; compare contrast)
        fill_to_border_contrast = float(np.linalg.norm(border_bgr - fill_bgr))
        border_to_bg_contrast = float(np.linalg.norm(bg_bgr - border_bgr))
        if fill_to_border_contrast >= 0.7 * border_to_bg_contrast:
            edge_start_idx = 0
            edge_end_idx = min(n - 1, border_center_idx + half_w)
        else:
            edge_start_idx = max(0, border_center_idx - half_w)
            edge_end_idx = n - 1
    else:
        # No separate outline stroke: locate peak boundary gradient point along P1->P2
        peak_idx = i_lo + int(np.argmax(grad[i_lo:i_hi]))
        # Sample just on the outer shoulder of the peak boundary change, or peak itself
        shoulder_idx = min(n - 1, peak_idx + max(1, int(round(0.03 * n))))
        b0 = max(0, shoulder_idx - 1)
        b1 = min(n, shoulder_idx + 2)
        border_bgr = colors[b0:b1].mean(axis=0)
        edge_start_idx = 0
        edge_end_idx = n - 1

    return border_bgr, has_distinct_border, edge_start_idx, edge_end_idx


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z_clipped = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _fit_sigmoid_cdf(d: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit f(d; d0, sw) = 1 / (1 + exp(-(d - d0) / sw)) to normalized edge profile y in [0, 1].
    Uses a fast coarse grid search followed by Gauss-Newton nonlinear least squares.

    Returns:
        (d0, sw): Inflection center (px) and scale width (px).
    """
    d_min = float(d[0])
    d_max = float(d[-1])
    span = max(d_max - d_min, 1.0)

    # Initial d0 estimate: where y crosses 0.5
    cross_idx = int(np.argmin(np.abs(y - 0.5)))
    d0_init = float(d[cross_idx])

    # Coarse grid search
    d0_grid = np.linspace(
        max(d_min, d0_init - 0.25 * span),
        min(d_max, d0_init + 0.25 * span),
        25,
    )
    sw_grid = np.geomspace(0.08, max(2.0, 0.35 * span), 24)

    best_sse = float("inf")
    best_d0 = d0_init
    best_sw = 0.4

    for sw_cand in sw_grid:
        # Vectorize across d0_grid
        z = (d[None, :] - d0_grid[:, None]) / sw_cand
        pred = _sigmoid(z)
        sse_vec = np.sum((pred - y[None, :]) ** 2, axis=1)
        idx = int(np.argmin(sse_vec))
        if float(sse_vec[idx]) < best_sse:
            best_sse = float(sse_vec[idx])
            best_d0 = float(d0_grid[idx])
            best_sw = float(sw_cand)

    # Gauss-Newton refinement (Levenberg-Marquardt damping)
    d0 = best_d0
    sw = best_sw
    lam = 1e-3

    for _ in range(25):
        z = (d - d0) / sw
        f = _sigmoid(z)
        res = f - y  # residual r_i = f_i - y_i
        sse = float(np.dot(res, res))

        df_dz = f * (1.0 - f)
        # Derivatives wrt (d0, sw)
        J0 = df_dz * (-1.0 / sw)
        J1 = df_dz * (-(d - d0) / (sw * sw))
        J = np.column_stack([J0, J1])

        JTJ = J.T @ J + lam * np.eye(2)
        JTr = J.T @ res
        try:
            step = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            break

        if float(np.linalg.norm(step)) < 1e-5:
            break

        d0_new = float(np.clip(d0 + step[0], d_min, d_max))
        sw_new = float(np.clip(sw + step[1], 0.05, span))
        f_new = _sigmoid((d - d0_new) / sw_new)
        sse_new = float(np.sum((f_new - y) ** 2))

        if sse_new < sse:
            d0, sw = d0_new, sw_new
            lam = max(1e-6, lam * 0.5)
        else:
            lam = min(1e3, lam * 5.0)

    return d0, sw


def estimate_edge_blur(
    dists: np.ndarray,
    colors: np.ndarray,
    edge_start_idx: int,
    edge_end_idx: int,
    scale_y: float,
) -> Tuple[float, float]:
    """
    Extract the 1D normalized transition profile along the edge, fit a
    Sigmoid/Gaussian CDF slope, compute the 10%->90% edge falloff width in pixels,
    and map it linearly to a libass \\blur value.

    Mapping calibration:
      - Sharp digital edge (w_10_90 <= 1.0 px) -> \\blur0.4
      - Soft/compressed edge (w_10_90 ~= 4.5 px) -> \\blur1.2

    Returns:
        (blur_val, falloff_width_px)
    """
    sub_d = dists[edge_start_idx : edge_end_idx + 1]
    sub_c = colors[edge_start_idx : edge_end_idx + 1]
    if len(sub_d) < 6:
        return 0.6, 1.875

    # Determine stable start and end colors from the outer 15% ends of the slice
    k = max(1, int(round(0.12 * len(sub_c))))
    c_start = sub_c[:k].mean(axis=0)
    c_end = sub_c[-k:].mean(axis=0)
    delta = c_end - c_start
    delta_norm = float(np.linalg.norm(delta))

    if delta_norm < 5.0:
        # Low contrast edge; fall back to luminance contrast or default
        return 0.6, 1.875

    u = delta / delta_norm
    proj = (sub_c - c_start[None, :]) @ u
    y_norm = np.clip(proj / delta_norm, 0.0, 1.0)

    # Isolate the monotonic transition neighborhood around the steepest slope
    dy = np.gradient(y_norm, sub_d)
    peak_i = int(np.argmax(dy))
    # Expand window around peak_i until profile flattens near 0 and 1
    left_i = peak_i
    while left_i > 0 and y_norm[left_i] > 0.04 and dy[left_i - 1] > 0.01:
        left_i -= 1
    right_i = peak_i
    while right_i < len(sub_d) - 1 and y_norm[right_i] < 0.96 and dy[right_i + 1] > 0.01:
        right_i += 1

    # Add a small shoulder margin on both sides for stable asymptote fitting
    pad = max(2, int(round(0.1 * len(sub_d))))
    left_i = max(0, left_i - pad)
    right_i = min(len(sub_d) - 1, right_i + pad)

    fit_d = sub_d[left_i : right_i + 1]
    fit_y = y_norm[left_i : right_i + 1]
    if len(fit_d) < 5:
        fit_d = sub_d
        fit_y = y_norm

    _, sw = _fit_sigmoid_cdf(fit_d, fit_y)

    # Logistic 10% -> 90% transition width: w_10_90 = 2 * ln(9) * sw
    w_10_90_video_px = 2.0 * math.log(9.0) * sw
    # Adjust to script resolution scale if PlayRes differs from video resolution
    res_factor = (1.0 / scale_y) if scale_y > 0 else 1.0
    w_10_90_script_px = w_10_90_video_px * res_factor

    # Linear mapping: <= 1.0 px -> 0.4, 4.5 px -> 1.2
    blur_raw = 0.4 + (max(1.0, w_10_90_script_px) - 1.0) * (1.2 - 0.4) / (4.5 - 1.0)
    blur_clamped = float(np.clip(blur_raw, 0.4, 5.0))
    return round(blur_clamped, 2), round(w_10_90_video_px, 3)


def analyze_film_grain(
    frame: np.ndarray,
    px2: float,
    py2: float,
    radius: int,
    sx2: float,
    sy2: float,
) -> Dict[str, object]:
    """
    Analyze the standard deviation of detrended luminance in a static background
    patch around Point 2. If visible grain is detected, generate an automatic
    low-opacity \\alpha blend tag and a tiny semi-transparent tiled \\p1 noise vector.
    """
    h, w = frame.shape[:2]
    grain_r = max(radius * 2, 5)
    ix = int(round(px2))
    iy = int(round(py2))
    x0 = max(0, ix - grain_r)
    x1 = min(w, ix + grain_r + 1)
    y0 = max(0, iy - grain_r)
    y1 = min(h, iy + grain_r + 1)

    patch = frame[y0:y1, x0:x1].astype(np.float64)
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return {
            "grain_sigma": 0.0,
            "grain_detected": False,
            "grain_alpha": "",
            "grain_vector": "",
        }

    luma = bgr_to_luma(patch)
    ph, pw = luma.shape

    # Fit a 2D plane z = a*x + b*y + c to remove smooth background shading/gradients
    yy, xx = np.mgrid[0:ph, 0:pw]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(ph * pw, dtype=np.float64)])
    coeffs, _, _, _ = np.linalg.lstsq(A, luma.ravel(), rcond=None)
    plane = (A @ coeffs).reshape(ph, pw)
    residual = luma - plane

    sigma_grain = float(np.std(residual))

    # Threshold for visible film grain (below 1.5 luma stddev is clean/flat digital)
    if sigma_grain < 1.5:
        return {
            "grain_sigma": round(sigma_grain, 3),
            "grain_detected": False,
            "grain_alpha": "",
            "grain_vector": "",
        }

    # Map sigma_grain in [1.5, 18.0] to subtle ASS alpha &H08&..&H30& (0=opaque, FF=transparent)
    norm_g = float(np.clip((sigma_grain - 1.5) / (18.0 - 1.5), 0.0, 1.0))
    alpha_val = int(round(0x08 + norm_g * (0x30 - 0x08)))
    grain_alpha = f"&H{alpha_val:02X}&"

    # Build a deterministic tiled micro-speckle \p1 vector overlay matched to grain intensity
    # Higher grain -> slightly more opaque mask (lower ASS 1a hex, e.g. &HE2& down to &HB8&)
    mask_alpha_val = int(round(0xE6 - norm_g * (0xE6 - 0xB8)))
    mask_alpha_hex = f"&H{mask_alpha_val:02X}&"

    bg_mean = patch.reshape(-1, 3).mean(axis=0)
    speckle_bgr = np.clip(bg_mean + 24.0, 0.0, 255.0)
    speckle_hex = bgr_to_ass_hex(speckle_bgr)

    # Generate a compact 24x24 tiled micro-speckle pattern in ASS vector coordinates
    rng = np.random.default_rng(42)
    draw_cmds = []
    n_speckles = int(round(10 + norm_g * 14))
    for _ in range(n_speckles):
        sx = int(rng.integers(0, 22))
        sy = int(rng.integers(0, 22))
        sw_box = int(rng.integers(1, 3))
        draw_cmds.append(
            f"m {sx} {sy} l {sx + sw_box} {sy} {sx + sw_box} {sy + sw_box} {sx} {sy + sw_box}"
        )
    drawing_str = " ".join(draw_cmds)
    pos_x = round(sx2, 1)
    pos_y = round(sy2, 1)
    grain_vector = (
        f"{{\\p1\\an7\\pos({pos_x},{pos_y})\\1c{speckle_hex}"
        f"\\1a{mask_alpha_hex}\\bord0\\shad0\\blur0.5}}{drawing_str}{{\\p0}}"
    )

    return {
        "grain_sigma": round(sigma_grain, 3),
        "grain_detected": True,
        "grain_alpha": grain_alpha,
        "grain_vector": grain_vector,
    }


def format_blur_number(blur: float) -> str:
    """Format blur cleanly (e.g. 0.4, 0.85, 1.2)."""
    s = f"{blur:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def run_worker(args: argparse.Namespace) -> Dict[str, object]:
    """Execute the full sampling & matching pipeline and return JSON-serializable dict."""
    radius = int(np.clip(args.radius, 1, 5))
    frame = load_frame(args.video, args.frame)
    frame_h, frame_w = frame.shape[:2]

    playresx = args.playresx if args.playresx > 0 else float(frame_w)
    playresy = args.playresy if args.playresy > 0 else float(frame_h)
    scale_y = frame_h / playresy if playresy > 0 else 1.0

    px1, py1 = scale_coords(args.x1, args.y1, playresx, playresy, frame_w, frame_h)
    px2, py2 = scale_coords(args.x2, args.y2, playresx, playresy, frame_w, frame_h)

    fill_bgr = sample_patch_bgr(frame, px1, py1, radius)
    bg_bgr = sample_patch_bgr(frame, px2, py2, radius)

    primary_hex = bgr_to_ass_hex(fill_bgr)
    bg_hex = bgr_to_ass_hex(bg_bgr)

    # Sample 1D ray using a thin perpendicular average so the edge profile isn't smeared
    perp_r = min(radius, 1)
    dists, colors, _ = sample_ray_profile(frame, (px1, py1), (px2, py2), perp_radius=perp_r)

    border_bgr, has_distinct_border, edge_start_idx, edge_end_idx = (
        detect_border_and_edge(dists, colors, fill_bgr, bg_bgr)
    )
    border_hex = bgr_to_ass_hex(border_bgr)

    blur_val, falloff_px = estimate_edge_blur(
        dists, colors, edge_start_idx, edge_end_idx, scale_y
    )

    if args.match_grain:
        grain_info = analyze_film_grain(frame, px2, py2, radius, args.x2, args.y2)
    else:
        grain_info = {
            "grain_sigma": 0.0,
            "grain_detected": False,
            "grain_alpha": "",
            "grain_vector": "",
        }

    # Build compiled ASS tags string according to requested targets
    tag_parts = []
    if args.match_primary:
        tag_parts.append(f"\\1c{primary_hex}")
    if args.match_border:
        tag_parts.append(f"\\3c{border_hex}")
    if args.match_blur:
        tag_parts.append(f"\\blur{format_blur_number(blur_val)}")
    if args.match_grain and grain_info["grain_detected"] and grain_info["grain_alpha"]:
        tag_parts.append(f"\\alpha{grain_info['grain_alpha']}")

    ass_tags = "{" + "".join(tag_parts) + "}" if tag_parts else ""

    return {
        "ok": True,
        "frame": int(args.frame),
        "video_resolution": [int(frame_w), int(frame_h)],
        "primary_color": primary_hex,
        "border_color": border_hex,
        "background_color": bg_hex,
        "has_distinct_border": bool(has_distinct_border),
        "blur": blur_val,
        "edge_falloff_px": falloff_px,
        "grain_sigma": grain_info["grain_sigma"],
        "grain_detected": grain_info["grain_detected"],
        "grain_alpha": grain_info["grain_alpha"],
        "grain_vector": grain_info["grain_vector"],
        "ass_tags": ass_tags,
    }


def main(argv: Optional[list[str]] = None) -> int:
    try:
        args = parse_args(argv)
        result = run_worker(args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
