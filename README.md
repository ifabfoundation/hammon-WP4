# HaMMon WP4 — Building Feature Extraction Pipeline

**Work Package 4 — Task 4.5 & Task 4.6**  
**IFAB Foundation**  
**Project Period: January 2024 – January 2026**

---

## Overview

Automated pipeline for extracting building structural features from multi-source geospatial data, combining Street View panoramic imagery processing with airborne LiDAR point cloud analysis. Produces georeferenced building attribute datasets for seismic and flood vulnerability assessment.

The pipeline implements classical computer vision methods — vanishing point detection, line segment analysis, and percentile-based height estimation — providing interpretable, deterministic, and reproducible results suitable for structural engineering applications.

---

## Repository Structure

```
hammon-WP4/
├── src/                          # Core processing modules
│   ├── common/                   #   Shared utilities (S3, config, logging, progress)
│   ├── rectification/            #   Module 1: panorama rectification (VP detection)
│   ├── cropping/                 #   Module 2: facade extraction + sky removal
│   └── lidar/                    #   Module 3: LiDAR height extraction
│
├── scripts/                      # SLURM scripts for HPC execution
│   ├── slurm_download_panoramas.sh       # Step 1: download panoramas from S3
│   ├── slurm_rectification_parallel.sh   # Step 2: rectify (4 parallel batches)
│   ├── launch_rectification_parallel.sh  # Step 2: launcher wrapper
│   ├── slurm_progressive_upload.sh       # Step 2b: progressive S3 upload
│   ├── slurm_cropping.sh                # Step 3: crop facades
│   ├── sync_crops_to_s3.sh              # Step 3b: sync crops to S3
│   ├── slurm_lidar_28zones.sh           # Step 4: LiDAR heights (28 zone array)
│   ├── merge_lidar_results.sh           # Step 5: merge GeoJSON results
│   └── archive/                          # Historical/test script variants
│
├── tools/                        # Standalone utility scripts
│   ├── download_s3_streetview.py         # Bulk S3 download with resume
│   ├── check_download_completeness.py    # S3 vs local diff check
│   └── transfer_s3_to_sharepoint.py      # S3 → SharePoint transfer (self-contained)
│
├── config.yaml                   # Pipeline configuration
├── requirements.txt              # Python dependencies
└── .gitignore
```

---

## Pipeline Modules

### 1. Panorama Rectification (`src/rectification/`)

Transforms 360° equirectangular Street View panoramas into rectified planar perspective views aligned with building facades.

- **Vanishing point detection** via Simon's method + LSD line segments + RANSAC consensus
- **Perspective transformation** with configurable FOV (154° horizontal)
- Output: rectified JPEG images + heading maps (JSON/NPY)

### 2. Facade Extraction (`src/cropping/`)

Extracts individual building facades from rectified images using coordinate-based cropping from a GeoDataFrame.

- **Coordinate-based bounding box** computation from geographic coordinates → pixel space
- **Automatic sky removal** via vertical gradient analysis with Gaussian smoothing
- Quality filtering: minimum width ≥ 30 px, height ≥ 100 px

### 3. LiDAR Height Extraction (`src/lidar/`)

Computes building heights from classified airborne LiDAR data (ASPRS LAS 1.4).

- **Cylindrical spatial query** with multi-radius fallback (3 m → 5 m → 8 m)
- Height = P95(building points) − P5(ground points)
- Minimum point thresholds with automatic class fallback
- Output: GeoJSON with height attributes and quality flags per building

---

## Requirements

- Python 3.12+
- Dependencies: `pip install -r requirements.txt`
- HPC with SLURM (for batch processing)
- S3-compatible object storage for data I/O

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set S3 credentials
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"

# Run LiDAR extraction for one zone
python -m src.lidar.batch_runner --zone centro1 --output heights.geojson

# Run rectification
python -m src.rectification.panorama_processor --config config.yaml
```

---

## HPC Deployment

The pipeline was designed for execution on SLURM-managed HPC clusters. The canonical execution order is:

1. `slurm_download_panoramas.sh` — download Street View panoramas from S3
2. `launch_rectification_parallel.sh` — rectify panoramas (4 parallel array jobs)
3. `slurm_progressive_upload.sh` — upload rectification results during processing
4. `slurm_cropping.sh` — extract building facades with sky removal
5. `sync_crops_to_s3.sh` — sync crop results to S3
6. `slurm_lidar_28zones.sh` — compute building heights (28-zone array job)
7. `merge_lidar_results.sh` — merge per-zone GeoJSON into final dataset

See individual script headers in `scripts/` for SLURM resource requirements and parameters.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| OpenCV (`cv2`) | Image processing and geometric transformations |
| NumPy | Numerical computing |
| GeoPandas | Geospatial data handling |
| laspy | LAS/LAZ point cloud I/O |
| scikit-image | Image analysis |
| pylsd-nova | Line Segment Detection |
| boto3 | S3-compatible object storage |

---

## License

This project is licensed under the [MIT License](LICENSE).
