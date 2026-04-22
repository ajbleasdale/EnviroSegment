"""
Quality control script for camera trap image datasets.

This script screens images for several common quality issues, including:
- blur
- obscuration / low-information frames
- near-infrared (NIR) imagery
- low-saturation winter scenes
- deep snow cover
- blue-dominated snow scenes
- snow-related low-texture blockage

Images that fail one or more enabled quality control checks are moved to a
'Quality_Control' subfolder within the root directory. The script also reports
summary counts for each enabled detector and each assigned failure reason.

Designed for batch processing of camera trap datasets stored in nested folders.
"""

import os
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np

# =============================================================================
# CONFIG / TOGGLES
# =============================================================================
RUN_NIR_DETECTION = True
RUN_BLUR_DETECTION = True
RUN_OBSCURED_DETECTION = True
RUN_LOW_SATURATION_DETECTION = True
RUN_DEEP_SNOW_DETECTION = True
RUN_BLUE_SNOW_DETECTION = True
RUN_BLOCKAGE_LOW_TEXTURE = True

SKIP_QUALITY_CONTROL_FOLDER = True
PRINT_CLASSIFICATIONS = True

QC_FOLDER_NAME = "Quality_Control" #creates folder within root directory
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# =============================================================================
# SMALL UTILITIES
# =============================================================================
def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _read_image(path: str):
    """Read an image preserving original channel structure. Returns None if unreadable."""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def _collision_safe_dest(dest_dir: str, filename: str) -> str:
    """If filename exists in dest_dir, append __{i} before extension."""
    base, ext = os.path.splitext(filename)
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        return dest_path

    i = 1
    while True:
        candidate = os.path.join(dest_dir, f"{base}__{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


# =============================================================================
# DETECTORS
# =============================================================================
def detect_nir(
    image,
    strip_width: int = 500,
    sat_threshold: int = 5,
    pink_hue_low_1: int = 140,
    pink_hue_high_1: int = 179,
    pink_hue_low_2: int = 0,
    pink_hue_high_2: int = 10,
    pink_sat_min: int = 40,
    pink_val_min: int = 40,
    pink_ratio_threshold: float = 0.35,
):
    """
    Identify NIR-like frames.

    Returns: (is_nir: bool, reason: str|None)

    Categories:
      - 2D grayscale array -> NIR_grayscale
      - center strip extremely low saturation -> NIR_low_saturation
      - image heavily pink/magenta (false-colour NIR style) -> NIR_pink_false_colour
    """
    if image is None:
        return False, None

    # True grayscale image
    if len(image.shape) == 2:
        return True, "NIR_grayscale"

    # Handle unexpected single-channel 3D arrays
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True, "NIR_grayscale"

    h, w = image.shape[:2]

    center_y = h // 2
    start_x = max(0, (w - strip_width) // 2)
    end_x = min(w, start_x + strip_width)

    center_strip = image[center_y:center_y + 1, start_x:end_x]
    hsv_strip = cv2.cvtColor(center_strip, cv2.COLOR_BGR2HSV)
    sat = hsv_strip[:, :, 1]

    if not (sat > sat_threshold).any():
        return True, "NIR_low_saturation"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)

    pink_mask = (
        (
            ((hh >= pink_hue_low_1) & (hh <= pink_hue_high_1))
            | ((hh >= pink_hue_low_2) & (hh <= pink_hue_high_2))
        )
        & (ss >= pink_sat_min)
        & (vv >= pink_val_min)
    )

    pink_ratio = float(np.mean(pink_mask)) if pink_mask.size else 0.0

    if pink_ratio >= pink_ratio_threshold:
        return True, "NIR_pink_false_colour"

    return False, None


def detect_blur(
    image,
    threshold: float = 5000.0,
    top_frac: float = 0.4,
    min_edge_ratio: float = 0.001,
    # Fallback: edge-poor + low contrast -> blurry
    low_edge_ratio: float = 0.001,
    low_contrast_std: float = 10.0,
    return_score: bool = False,
):
    """
    Edge-masked Tenengrad focus on top ROI.
    Returns bool, or (bool, score) if return_score=True.
    """
    if image is None:
        return (False, 0.0) if return_score else False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    roi = gray[: int(h * top_frac), :]

    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(roi_blur, 50, 150)
    edge_ratio = float(np.mean(edges > 0))

    # Fallback: almost no edges + very low contrast (fog/smear/covered lens)
    roi_std = float(np.std(roi))
    if edge_ratio < low_edge_ratio:
        blurry = roi_std < low_contrast_std
        return (blurry, roi_std) if return_score else blurry

    _ = min_edge_ratio

    gx = cv2.Sobel(roi_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_blur, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy

    score = float(np.mean(energy[edges > 0])) if edge_ratio > 0 else 0.0
    blurry = score < threshold

    return (blurry, score) if return_score else blurry


def detect_obscuration(
    image,
    roi: str = "full",   # "full" or "top"
    top_frac: float = 0.25,
    max_edge_ratio: float = 0.05,
    max_std: float = 50.0,
    min_mean: float = 50.0,
    return_score: bool = False,
):
    """
    Low-information frame detector: fog/condensation/covered lens.
    obscured = (edge_ratio <= max_edge_ratio) AND (std <= max_std)
    Skips dark frames (mean < min_mean).
    """
    if image is None:
        return (False, {}) if return_score else False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if roi == "top":
        h = gray.shape[0]
        gray = gray[: int(h * top_frac), :]

    mean = float(np.mean(gray))
    std = float(np.std(gray))
    if mean < min_mean:
        if return_score:
            return False, {"mean": mean, "std": std, "edge_ratio": None}
        return False

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 50, 150)
    edge_ratio = float(np.mean(edges > 0))

    obscured = (edge_ratio <= max_edge_ratio) and (std <= max_std)

    if return_score:
        return obscured, {"mean": mean, "std": std, "edge_ratio": edge_ratio}
    return obscured


def detect_snow_low_saturation(
    image,
    saturation_threshold: int = 60,
    low_sat_ratio_threshold: float = 0.75,
    green_hue_range=(30, 90),
    green_allowance: float = 0.25,
) -> bool:
    """Low-saturation (gray) winter scenes, unless low-sat pixels contain substantial green hues."""
    if image is None:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, ss, _ = cv2.split(hsv)

    low_sat = ss < saturation_threshold
    low_sat_ratio = float(np.sum(low_sat) / ss.size) if ss.size else 0.0
    if low_sat_ratio < low_sat_ratio_threshold:
        return False

    if np.sum(low_sat) == 0:
        return False

    green_hue = (hh >= green_hue_range[0]) & (hh <= green_hue_range[1])
    green_in_low = green_hue & low_sat
    green_ratio_in_low = float(np.sum(green_in_low) / np.sum(low_sat))

    return green_ratio_in_low < green_allowance


def detect_snow_covered_bottom(
    image,
    std_threshold: float = 12,
    flat_row_std: float = 12,
    flat_row_ratio: float = 0.5,
    min_brightness: float = 50,
    green_ratio_threshold: float = 0.25,
    green_hue_range=(30, 90),
    green_min_saturation: int = 40,
) -> bool:
    """Deep snow flatness on lower half; skips if too green overall."""
    if image is None:
        return False

    h = image.shape[0]
    bottom = image[int(h * 0.5):, :, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)

    std_dev = float(np.std(gray))
    mean_brightness = float(np.mean(gray))

    row_std = np.std(gray, axis=1)
    flat_rows_ratio = float(np.sum(row_std < flat_row_std) / len(row_std))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, ss, _ = cv2.split(hsv)

    green_mask = (
        (hh >= green_hue_range[0])
        & (hh <= green_hue_range[1])
        & (ss >= green_min_saturation)
    )
    green_ratio = float(np.mean(green_mask))
    if green_ratio > green_ratio_threshold:
        return False

    return (std_dev < std_threshold) and (flat_rows_ratio > flat_row_ratio) and (mean_brightness > min_brightness)


def detect_snow_blue_dominated(
    image,
    blue_ratio_threshold: float = 0.6,
    green_hue_range=(30, 90),
    green_top_threshold: float = 0.25,
    verbose: bool = False,
) -> bool:
    """Blue-dominant pixels in bottom half indicate heavy snow; skip if top half is too green."""
    if image is None:
        return False

    h = image.shape[0]
    bottom = image[int(h * 0.5):, :, :]
    b, g, r = cv2.split(bottom)

    blue_mask = (b > r + 20) & (b > g + 20)
    blue_ratio = float(np.sum(blue_mask) / blue_mask.size) if blue_mask.size else 0.0

    if verbose:
        print(f"Blue ratio (bottom 50%): {blue_ratio:.3f}")

    if blue_ratio <= blue_ratio_threshold:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    top = hsv[: h // 2, :, :]
    hh, ss, _ = cv2.split(top)

    green_mask = (hh >= green_hue_range[0]) & (hh <= green_hue_range[1]) & (ss > 30)
    green_ratio_top = float(np.sum(green_mask) / green_mask.size) if green_mask.size else 0.0

    if verbose:
        print(f"Green ratio (top half): {green_ratio_top:.3f}")

    return green_ratio_top <= green_top_threshold


def _tenengrad_score(gray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def detect_snow_blockage(
    image,
    bottom_frac: float = 0.30,
    top_frac: float = 0.30,
    blur_ksize: int = 5,
    bottom_abs_max: float = 1200.0,
    drop_ratio_max: float = 0.5,
    green_hue_range=(30, 90),
    green_min_saturation: int = 35,
    green_ratio_full_min: float = 0.40,
    green_ratio_top_min: float = 0.50,
) -> bool:
    """
    Snow-only low-texture blockage detector:
      - skip if strongly green
      - apply only if bottom is snow-like (neutral white/grey or blue-tinted snow)
      - then flag if bottom is low texture + much lower than top
    """
    if image is None:
        return False

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hh, ss, _ = cv2.split(hsv)

    green_mask_full = (
        (hh >= green_hue_range[0])
        & (hh <= green_hue_range[1])
        & (ss >= green_min_saturation)
    )
    green_ratio_full = float(np.mean(green_mask_full))
    H = image.shape[0]
    green_ratio_top = float(np.mean(green_mask_full[: H // 2, :]))

    if (green_ratio_full >= green_ratio_full_min) or (green_ratio_top >= green_ratio_top_min):
        return False

    bottom_bgr = image[int(H * (1 - bottom_frac)):, :, :]
    hsv_bottom = cv2.cvtColor(bottom_bgr, cv2.COLOR_BGR2HSV)
    _, s_bot, v_bot = cv2.split(hsv_bottom)
    b, g, r = cv2.split(bottom_bgr)

    neutral = (np.abs(r - g) < 20) & (np.abs(r - b) < 20) & (np.abs(g - b) < 20)
    snow_neutral = (v_bot >= 40) & (s_bot <= 120) & neutral
    snow_blue = (b > r + 15) & (b > g + 15) & (v_bot >= 120)

    snow_ratio = float(np.mean(snow_neutral | snow_blue))
    if snow_ratio < 0.20:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bottom = gray[int(H * (1 - bottom_frac)):, :]
    top = gray[: int(H * top_frac), :]

    bottom = cv2.GaussianBlur(bottom, (blur_ksize, blur_ksize), 0)
    top = cv2.GaussianBlur(top, (blur_ksize, blur_ksize), 0)

    score_bottom = _tenengrad_score(bottom)
    score_top = _tenengrad_score(top)

    if score_top < 1e-6:
        return False

    ratio = score_bottom / score_top
    return (score_bottom <= bottom_abs_max) and (ratio <= drop_ratio_max)


# =============================================================================
# QC DRIVER
# =============================================================================
def run_quality_control(root_dir: str):
    qc_dir = os.path.join(root_dir, QC_FOLDER_NAME)
    _safe_makedirs(qc_dir)

    moved_total = 0
    moved_by_reason = Counter()
    moved_by_primary_reason = Counter()
    moved_by_toggle = Counter()
    moved_images_reasons = defaultdict(list)

    reason_to_toggle = {
        "NIR_grayscale": "RUN_NIR_DETECTION",
        "NIR_low_saturation": "RUN_NIR_DETECTION",
        "NIR_pink_false_colour": "RUN_NIR_DETECTION",
        "blur": "RUN_BLUR_DETECTION",
        "obscured": "RUN_OBSCURED_DETECTION",
        "low_saturation": "RUN_LOW_SATURATION_DETECTION",
        "snow_blockage": "RUN_DEEP_SNOW_DETECTION",
        "blue_snow": "RUN_BLUE_SNOW_DETECTION",
        "blockage_low_texture": "RUN_BLOCKAGE_LOW_TEXTURE",
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if SKIP_QUALITY_CONTROL_FOLDER and QC_FOLDER_NAME in dirnames:
            dirnames.remove(QC_FOLDER_NAME)

        for fname in filenames:
            if not _is_image_file(fname):
                continue

            src_path = os.path.join(dirpath, fname)

            if os.path.commonpath([src_path, qc_dir]) == qc_dir:
                continue

            image = _read_image(src_path)
            if image is None:
                continue

            reasons = []

            # 1) NIR
            if RUN_NIR_DETECTION:
                is_nir, nir_reason = detect_nir(image)
                if is_nir:
                    reasons.append(nir_reason)
                    if PRINT_CLASSIFICATIONS:
                        print(f"[NIR] {nir_reason}: {src_path}")

            # 2) Blur
            if RUN_BLUR_DETECTION and detect_blur(image):
                reasons.append("blur")
                if PRINT_CLASSIFICATIONS:
                    print(f"[BLUR] blur: {src_path}")

            # 3) Obscured
            if RUN_OBSCURED_DETECTION and detect_obscuration(image):
                reasons.append("obscured")
                if PRINT_CLASSIFICATIONS:
                    print(f"[OBSCURED] obscured: {src_path}")

            # 4) Low saturation
            if RUN_LOW_SATURATION_DETECTION and detect_snow_low_saturation(image):
                reasons.append("low_saturation")
                if PRINT_CLASSIFICATIONS:
                    print(f"[LOW_SAT] low_saturation: {src_path}")

            # 5) Deep snow blockage
            if RUN_DEEP_SNOW_DETECTION and detect_snow_covered_bottom(image):
                reasons.append("snow_blockage")
                if PRINT_CLASSIFICATIONS:
                    print(f"[SNOW] snow_blockage: {src_path}")

            # 6) Blue snow
            if RUN_BLUE_SNOW_DETECTION and detect_snow_blue_dominated(image):
                reasons.append("blue_snow")
                if PRINT_CLASSIFICATIONS:
                    print(f"[BLUE_SNOW] blue_snow: {src_path}")

            # 7) Snow-only low-texture blockage
            if RUN_BLOCKAGE_LOW_TEXTURE and detect_snow_blockage(image):
                reasons.append("blockage_low_texture")
                if PRINT_CLASSIFICATIONS:
                    print(f"[BLOCKAGE] blockage_low_texture: {src_path}")

            if not reasons:
                continue

            moved_total += 1
            moved_by_reason.update(reasons)
            moved_by_primary_reason.update([reasons[0]])
            moved_images_reasons[src_path] = reasons

            for r in reasons:
                moved_by_toggle[reason_to_toggle.get(r, r)] += 1

            dest_path = _collision_safe_dest(qc_dir, fname)
            shutil.move(src_path, dest_path)

    _print_summary(moved_total, moved_by_toggle, moved_by_reason, moved_by_primary_reason)

    return {
        "moved_total": moved_total,
        "moved_by_toggle": dict(moved_by_toggle),
        "moved_by_reason": dict(moved_by_reason),
        "moved_by_primary_reason": dict(moved_by_primary_reason),
        "moved_images_reasons": dict(moved_images_reasons),
        "quality_control_dir": qc_dir,
    }


def _print_summary(moved_total, moved_by_toggle, moved_by_reason, moved_by_primary_reason):
    toggles_in_order = [
        ("RUN_NIR_DETECTION", RUN_NIR_DETECTION),
        ("RUN_BLUR_DETECTION", RUN_BLUR_DETECTION),
        ("RUN_OBSCURED_DETECTION", RUN_OBSCURED_DETECTION),
        ("RUN_LOW_SATURATION_DETECTION", RUN_LOW_SATURATION_DETECTION),
        ("RUN_DEEP_SNOW_DETECTION", RUN_DEEP_SNOW_DETECTION),
        ("RUN_BLUE_SNOW_DETECTION", RUN_BLUE_SNOW_DETECTION),
        ("RUN_BLOCKAGE_LOW_TEXTURE", RUN_BLOCKAGE_LOW_TEXTURE),
    ]

    print("\n=== QC SUMMARY ===")
    print(f"Total images moved: {moved_total}\n")

    print("{:<30} {:<8} {:>10}".format("Toggle", "Enabled", "Moved"))
    print("-" * 52)
    for tname, enabled in toggles_in_order:
        print("{:<30} {:<8} {:>10}".format(tname, str(enabled), moved_by_toggle.get(tname, 0)))

    print("\n=== REASONS (multi-label counts) ===")
    for reason, cnt in moved_by_reason.most_common():
        print(f"{reason:>22}: {cnt}")

    print("\n=== PRIMARY REASON (first hit per moved image) ===")
    for reason, cnt in moved_by_primary_reason.most_common():
        print(f"{reason:>22}: {cnt}")


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    ROOT_DIR = r"link-to-root-directory"
    run_quality_control(ROOT_DIR)
