# Script to copy images from a source directory based on filename matches in a second directory.
# Only images with matching filenames (ignoring file extension and case) are copied.
#
# Used to subset images (original images or masks) based on a selected set (annotated overlay outputs).

import os
import shutil

# ======================
# PATHS (EDIT THESE)
# ======================
source_folder = r"link-to-source-folder"  # files to copy FROM
match_folder = r"link-to-truth-folder"  # files to match against
output_folder = r"link-toi-new-directory"  # files copied TO

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# ======================
# SETTINGS
# ======================
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ======================
# GET STEMS FROM MATCH FOLDER
# ======================
# Build a set of filenames (without extensions) to match against
match_stems = set()

for f in os.listdir(match_folder):
    if f.lower().endswith(IMAGE_EXTENSIONS):
        stem = os.path.splitext(f)[0].lower()
        match_stems.add(stem)

# ======================
# COPY MATCHING FILES
# ======================
copied = 0
skipped = 0

for filename in os.listdir(source_folder):
    source_path = os.path.join(source_folder, filename)

    if not os.path.isfile(source_path):
        continue

    if not filename.lower().endswith(IMAGE_EXTENSIONS):
        continue

    stem = os.path.splitext(filename)[0].lower()

    # Copy file if stem matches
    if stem in match_stems:
        dest_path = os.path.join(output_folder, filename)
        shutil.copy2(source_path, dest_path)
        copied += 1
    else:
        skipped += 1

print(f"Copied: {copied}")
print(f"Skipped (no match): {skipped}")
