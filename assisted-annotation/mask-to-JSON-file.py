# Script to convert predicted segmentation masks into Label Studio JSON tasks.
# Reads image and mask filenames from a CSV file and generates polygon annotations
# for each predicted class.
#
# Used to pre-load model predictions into Label Studio for human-in-the-loop annotation.

import os
import json
import cv2
import pandas as pd
import numpy as np

# ==========================
# CONFIG (EDIT THESE)
# ==========================
CSV_PATH = r"link-to-csv-file"

IMAGE_COL = "image_name"       # Column containing image filenames
MASK_COL = "mask_name"         # Column containing mask filenames; set to None if masks use image name
SUBFOLDER_COL = "subfolder"    # Column containing image subfolder names (e.g. SE-NM-1)

IMAGES_DIR = r"link-to-image-directory"
MASKS_DIR = r"link-to-mask-directory"

IMAGE_BASE_URL = "http://127.0.0.1:8001"  # Base URL used by Label Studio to access images
OUT_JSON = r"name-outoutput-JSON-file"

# These must match the tag names in the Label Studio labeling configuration
FROM_NAME = "label"   # <PolygonLabels name="label" ...>
TO_NAME = "image"     # <Image name="image" ...>

# ==========================
# CLASS MAP
# ==========================
# Class names and corresponding pixel values in the mask
class_map = {
    "background": 0,
    "sky": 1,
    "deadwood": 2,
    "bare": 3,
    "rock": 4,
    "snow_ice": 5,

    "low_lying_shrub": 6,
    "briar_shrub": 7,

    "graminoid": 8,
    "forb": 9,
    "fern": 10,

    "bryophyte": 11,
    "lichen": 12,

    "conifer": 13,
    "broadleaf_deciduous": 14,
    "broadleaf_evergreen": 15,
    "vine_evergreen": 16,

    "leaf_litter": 17,
    "deciduous_winter": 18,
    "broadleaf_seedling": 19,

    "ignore": 255,
}

# Include only non-background, non-ignore classes in predictions
PREDICT_CLASSES = [k for k, v in class_map.items() if v not in (0, 255)]

# ==========================
# POLYGON EXTRACTION SETTINGS
# ==========================
MIN_AREA_PX = 50
APPROX_EPS_FRAC = 0.0005

# Overlap, boundary, and splitting controls
NO_OVERLAP = True     # Prevent overlapping polygons between classes
ERODE_PX = 1          # Shrink class regions slightly to reduce boundary overlap
SPLIT_OPEN_PX = 2     # Apply morphological opening to split large connected regions

# ==========================
# HELPERS
# ==========================
def file_exists(p: str) -> bool:
    """Return True if file exists."""
    try:
        return os.path.isfile(p)
    except OSError:
        return False


def make_image_url(subfolder: str, fname: str) -> str:
    """Build image URL for Label Studio."""
    return f"{IMAGE_BASE_URL}/{subfolder}/{fname}"


def find_pred_mask_path(subfolder: str, img_filename: str, mask_name: str | None):
    """
    Find the predicted mask file.

    Checks both:
    1. Flat mask directory
    2. Per-subfolder mask directory
    """
    base, _ = os.path.splitext(img_filename)

    candidates = []
    if mask_name:
        candidates.append(mask_name)
    candidates.append(f"{base}.png")
    candidates.append(f"{img_filename}.png")

    # Try flat mask folder
    for cand in candidates:
        p = os.path.join(MASKS_DIR, cand)
        if file_exists(p):
            return p

    # Try subfolder structure
    for cand in candidates:
        p = os.path.join(MASKS_DIR, subfolder, cand)
        if file_exists(p):
            return p

    raise FileNotFoundError(f"No predicted mask found for {subfolder}/{img_filename} (tried: {candidates})")


def _apply_erode_and_split(bw_255: np.ndarray) -> np.ndarray:
    """
    Apply optional morphological opening and erosion to a binary mask.
    Input must be uint8 with values {0,255}.
    """
    out = bw_255

    # Split large blobs and remove thin bridges
    if SPLIT_OPEN_PX > 0:
        k = 2 * SPLIT_OPEN_PX + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)

    # Shrink boundaries slightly
    if ERODE_PX > 0:
        k = 2 * ERODE_PX + 1
        kernel = np.ones((k, k), np.uint8)
        out = cv2.erode(out, kernel, iterations=1)

    return out


def mask_to_polygons_by_class(mask_path: str):
    """
    Convert a predicted mask into polygons grouped by class.

    Returns:
        dict[label_name] = list of (points_pct, width, height)

    Notes:
    - Background and ignore classes are excluded
    - Optional overlap prevention can be applied
    - Optional morphology can be used to improve polygon separation
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Could not read mask: {mask_path}")

    h, w = m.shape[:2]
    out = {k: [] for k in PREDICT_CLASSES}

    occupied = np.zeros((h, w), dtype=bool) if NO_OVERLAP else None

    for label_name in PREDICT_CLASSES:
        class_id = class_map[label_name]

        if NO_OVERLAP:
            bw_bool = (m == class_id) & (~occupied)
            occupied |= bw_bool
        else:
            bw_bool = (m == class_id)

        bw = bw_bool.astype(np.uint8) * 255
        bw = _apply_erode_and_split(bw)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA_PX:
                continue

            peri = cv2.arcLength(cnt, True)
            eps = max(1.0, APPROX_EPS_FRAC * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)

            pts = approx.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] < 3:
                continue

            pts_pct = []
            for x, y in pts:
                x_pct = float(np.clip((x / w) * 100.0, 0.0, 100.0))
                y_pct = float(np.clip((y / h) * 100.0, 0.0, 100.0))
                pts_pct.append([x_pct, y_pct])

            out[label_name].append((pts_pct, w, h))

    return out

# ==========================
# BUILD LABEL STUDIO TASKS
# ==========================
df = pd.read_csv(CSV_PATH)

tasks = []
missing_images = []
missing_pred_masks = []
made = 0

for _, row in df.iterrows():
    img_name = str(row[IMAGE_COL]).strip()
    subfolder = str(row[SUBFOLDER_COL]).strip()
    mask_name = None if MASK_COL is None else str(row[MASK_COL]).strip()

    # Check that the source image exists
    img_path = (
        os.path.join(IMAGES_DIR, subfolder, img_name)
        if os.path.isdir(os.path.join(IMAGES_DIR, subfolder))
        else os.path.join(IMAGES_DIR, img_name)
    )

    if not file_exists(img_path):
        missing_images.append(f"{subfolder}/{img_name}")
        continue

    # Find the corresponding predicted mask
    try:
        pred_mask_path = find_pred_mask_path(subfolder, img_name, mask_name)
    except FileNotFoundError:
        missing_pred_masks.append(f"{subfolder}/{img_name}")
        continue

    # Convert mask to polygons
    polys_by_class = mask_to_polygons_by_class(pred_mask_path)

    result = []
    for label_name, polygons in polys_by_class.items():
        for pts_pct, w, h in polygons:
            result.append({
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "polygonlabels",
                "value": {
                    "points": pts_pct,
                    "closed": True,
                    "polygonlabels": [label_name],
                },
                "original_width": w,
                "original_height": h,
                "image_rotation": 0
            })

    # Create Label Studio task with prediction result
    task = {
        "data": {
            "image": make_image_url(subfolder, img_name)
        },
        "predictions": [{
            "model_version": "model_pred_mask2poly",
            "score": 1.0,
            "result": result
        }]
    }

    tasks.append(task)
    made += 1

# ==========================
# SAVE OUTPUT
# ==========================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False)

print(f"Wrote: {OUT_JSON}")
print(f"Tasks created: {made}")
print(f"Missing images: {len(missing_images)}")
print(f"Missing predicted masks: {len(missing_pred_masks)}")

if missing_images[:10]:
    print("Example missing images:", missing_images[:10])

if missing_pred_masks[:10]:
    print("Example missing predicted masks:", missing_pred_masks[:10])
