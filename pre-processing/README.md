# EnviroSegment – Camera Trap Preprocessing Pipeline

The following steps outline the preprocessing workflow for camera trap imagery used in the EnviroSegment package.

## Overview

This pipeline standardises filenames, harmonises deployment IDs, and prepares images for downstream processing.

## Workflow
### Step 1 — Generate Image Date and Time Table

Run: list-dataset-date-time

Creates a CSV file containing:

image filename
acquisition date
acquisition time

This information is used to construct standardised filenames.

### Step 2 — Update Deployment IDs

Run: update-deployment-id

Requires a metadata file containing deployment IDs.

This step maps old deployment IDs to new standardised IDs
ensures consistency across datasets from multiple projects

### Step 3 — Rename Image Files

Run: file-rename

After constructing the new filenames in the CSV file, this step:

renames all images within the directory and preserves folder structure

### Step 4 — Crop Images

Run: cropping script

Removes camera-specific headers and footers from images.

## Important Notes
Renaming is performed before cropping to:
allow cross-referencing of filenames with camera records (date and time)
avoid potential issues caused by loss or modification of image metadata during processing
