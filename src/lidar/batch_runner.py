"""
Batch runner to extract building heights from segmented LiDAR LAS files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.common.config_loader import ConfigLoader
from src.common.logger import setup_logging
from src.common.s3_client import S3Client
from src.lidar.height_extractor import compute_height
from src.lidar.las_reader import LasRunLoader

logger = logging.getLogger(__name__)


def parse_radii(radii_arg: str) -> List[float]:
    return [float(r.strip()) for r in radii_arg.split(",") if r.strip()]


def load_geodataframe(s3_client: S3Client, bucket: str, key: str) -> gpd.GeoDataFrame:
    logger.info(f"Downloading GeoDataFrame from s3://{bucket}/{key}")
    gdf = s3_client.read_geodataframe(bucket, key)
    logger.info(f"GDF columns: {list(gdf.columns)}")
    return gdf


def resolve_column(desired: str, available: Iterable[str]) -> str:
    """
    Resolve a column name case-insensitively; return matching original name or raise.
    """
    available_list = list(available)
    lower_map = {c.lower(): c for c in available_list}
    if desired in available_list:
        return desired
    if desired.lower() in lower_map:
        resolved = lower_map[desired.lower()]
        logger.info(f"Resolved column '{desired}' to '{resolved}' (case-insensitive match)")
        return resolved
    raise KeyError(f"Column '{desired}' not found in GeoDataFrame. Available: {available_list}")


def process_run(
    run_loader: LasRunLoader,
    zone: str,
    run_id: str,
    buildings: pd.DataFrame,
    radii: Sequence[float],
    class_map: Dict[str, Iterable[int]],
    params: Dict[str, object],
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    """Download LAS for a run, load points in a bounding box, and compute heights per building."""
    max_radius = max(radii)
    minx = buildings[x_col].min() - max_radius
    maxx = buildings[x_col].max() + max_radius
    miny = buildings[y_col].min() - max_radius
    maxy = buildings[y_col].max() + max_radius
    bbox = (minx, miny, maxx, maxy)

    local_paths = run_loader.download_run_files(zone, run_id)
    class_hist = run_loader.summarize_classes(local_paths)
    logger.info(f"Class histogram for zone={zone} run={run_id}: {dict(class_hist)}")

    points = run_loader.load_points(local_paths, bbox=bbox)
    if len(points["x"]) == 0:
        logger.warning(f"No points loaded for zone={zone} run={run_id} within bbox {bbox}")
        buildings["height_m"] = np.nan
        buildings["ground_z"] = np.nan
        buildings["roof_z"] = np.nan
        buildings["radius_used"] = np.nan
        buildings["height_flags"] = "no_points_loaded"
        return buildings

    heights: List[Dict[str, object]] = []
    for _, row in buildings.iterrows():
        midpoint = (row[x_col], row[y_col])
        metrics = compute_height(
            midpoint=midpoint,
            points=points,
            radii=radii,
            class_map=class_map,
            ground_percentile=params["ground_percentile"],
            roof_percentile=params["roof_percentile"],
            min_ground_points=params["min_ground_points"],
            min_building_points=params["min_building_points"],
        )
        heights.append(metrics)

    # Attach results
    buildings = buildings.copy()
    buildings["ground_z"] = [m["ground_z"] for m in heights]
    buildings["roof_z"] = [m["roof_z"] for m in heights]
    buildings["height_m"] = [m["height_m"] for m in heights]
    buildings["radius_used"] = [m["radius_used"] for m in heights]
    buildings["height_flags"] = [";".join(m.get("flags", [])) for m in heights]
    buildings["height_class_counts"] = [json.dumps(m.get("counts", {})) for m in heights]
    return buildings


def main():
    parser = argparse.ArgumentParser(description="Extract building heights from segmented LiDAR.")
    parser.add_argument("--zone", type=str, default=None, help="Filter to a single zone")
    parser.add_argument("--runs", type=str, default=None, help="Comma-separated list of run ids to process")
    parser.add_argument("--gdf-bucket", type=str, default="ifab.data", help="Bucket for the building GeoDataFrame")
    parser.add_argument(
        "--gdf-key",
        type=str,
        default="geodataframe/facades_matched_complete.geojson",
        help="Key for the building GeoDataFrame",
    )
    parser.add_argument("--x-col", type=str, default="midpoint_x", help="Column for facade midpoint X")
    parser.add_argument("--y-col", type=str, default="midpoint_y", help="Column for facade midpoint Y")
    parser.add_argument("--zone-col", type=str, default="PATH", help="Column containing zone id")
    parser.add_argument("--run-col", type=str, default="RUN", help="Column containing run id (case-insensitive)")
    parser.add_argument("--output", type=str, default="outputs/lidar/heights.csv", help="Local output file path")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "geojson"],
        default="csv",
        help="Output format",
    )
    parser.add_argument("--radii", type=str, default="3,5,8", help="Comma-separated radii in meters")
    parser.add_argument("--ground-percentile", type=float, default=5.0, help="Percentile for ground elevation")
    parser.add_argument("--roof-percentile", type=float, default=95.0, help="Percentile for roof elevation")
    parser.add_argument("--min-ground-points", type=int, default=30, help="Minimum ground-class points before fallback")
    parser.add_argument("--min-building-points", type=int, default=30, help="Minimum building-class points before fallback")
    parser.add_argument(
        "--class-map",
        type=str,
        default="ground:2;building:6;vegetation:3,4,5",
        help="Class map specification like 'ground:2;building:6;vegetation:3,4,5'",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N buildings after filtering")
    parser.add_argument("--run-limit", type=int, default=None, help="Process only the first N (zone, run) groups")
    parser.add_argument("--temp-dir", type=str, default="tmp/lidar", help="Temporary directory for LAS downloads")
    parser.add_argument(
        "--output-fields",
        type=str,
        default=None,
        help="Comma-separated list of fields to write (geometry auto-added for GeoJSON if present)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        default=False,
        help="Upload output file to S3 after writing locally",
    )
    parser.add_argument(
        "--s3-output-bucket",
        type=str,
        default="ifab.data",
        help="S3 bucket for output upload",
    )
    parser.add_argument(
        "--s3-output-key",
        type=str,
        default=None,
        help="S3 key for output upload (default: lidar_heights/<filename>)",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    config_loader = ConfigLoader()
    s3_cfg = config_loader.get_s3_config()
    s3_client = S3Client(
        endpoint_url=s3_cfg.get("endpoint_url"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    radii = parse_radii(args.radii)

    class_map: Dict[str, List[int]] = {}
    for token in args.class_map.split(";"):
        if not token.strip():
            continue
        name, values = token.split(":")
        class_map[name.strip()] = [int(v) for v in values.split(",") if v]

    gdf = load_geodataframe(s3_client, args.gdf_bucket, args.gdf_key)
    logger.info(f"Loaded {len(gdf)} buildings from GeoDataFrame")

    # Resolve column names (case-insensitive)
    zone_col = resolve_column(args.zone_col, gdf.columns)
    run_col = resolve_column(args.run_col, gdf.columns)

    if args.zone:
        gdf = gdf[gdf[zone_col] == args.zone]
        logger.info(f"Filtered to zone={args.zone}, remaining {len(gdf)} rows")

    if args.runs:
        run_filter = set(r.strip() for r in args.runs.split(","))
        gdf = gdf[gdf[run_col].astype(str).isin(run_filter)]
        logger.info(f"Filtered to runs={run_filter}, remaining {len(gdf)} rows")

    # Sort by zone/run for deterministic grouping
    gdf = gdf.sort_values([zone_col, run_col])

    if args.limit is not None:
        gdf = gdf.head(args.limit)
        logger.info(f"Limiting to first {args.limit} rows, remaining {len(gdf)}")

    if args.x_col not in gdf.columns or args.y_col not in gdf.columns:
        raise ValueError(f"GeoDataFrame must contain columns {args.x_col} and {args.y_col}")

    run_loader = LasRunLoader(
        s3_client=s3_client,
        bucket=s3_cfg["input_bucket"],
        prefix=s3_cfg.get("input_prefix", "2025"),
        temp_dir=args.temp_dir,
    )

    processed_parts: List[pd.DataFrame] = []
    grouped = gdf.groupby([zone_col, run_col])
    total_groups = len(grouped)
    total_buildings = len(gdf)
    logger.info(f"Starting processing of {total_buildings} buildings across {total_groups} (zone, run) groups")
    processed_groups = 0
    processed_buildings = 0

    for (zone_val, run_val), group in grouped:
        if args.run_limit is not None and processed_groups >= args.run_limit:
            logger.info(f"Reached run-limit={args.run_limit}, stopping further processing.")
            break

        group_count = len(group)
        logger.info(
            f"[{processed_groups + 1}/{total_groups}] Processing zone={zone_val} run={run_val} "
            f"with {group_count} buildings"
        )

        processed = process_run(
            run_loader=run_loader,
            zone=str(zone_val),
            run_id=str(run_val),
            buildings=group,
            radii=radii,
            class_map=class_map,
            params={
                "ground_percentile": args.ground_percentile,
                "roof_percentile": args.roof_percentile,
                "min_ground_points": args.min_ground_points,
                "min_building_points": args.min_building_points,
            },
            x_col=args.x_col,
            y_col=args.y_col,
        )
        processed_parts.append(processed)
        processed_groups += 1
        processed_buildings += len(processed)
        logger.info(
            f"Completed zone={zone_val} run={run_val}; cumulative buildings processed: "
            f"{processed_buildings}/{total_buildings}"
        )

    if not processed_parts:
        logger.warning("No data processed; exiting without writing output.")
        return

    out_df = pd.concat(processed_parts, ignore_index=True)
    logger.info(f"Processed {len(out_df)} buildings (after filters/limits)")

    # Keep only selected fields if requested
    if args.output_fields:
        requested = [f.strip() for f in args.output_fields.split(",") if f.strip()]
        cols = []
        for col in requested:
            if col in out_df.columns:
                cols.append(col)
            else:
                logger.warning(f"Requested output field '{col}' not in data; skipping.")
        # Preserve geometry if present and not explicitly requested, for GeoJSON outputs
        if "geometry" in out_df.columns and "geometry" not in cols:
            cols.append("geometry")
        if cols:
            out_df = out_df[cols]
        else:
            logger.warning("No valid output fields requested; writing all columns.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == "geojson" and "geometry" in out_df.columns:
        crs = gdf.crs if isinstance(gdf, gpd.GeoDataFrame) else None
        gdf_out = gpd.GeoDataFrame(out_df, geometry="geometry", crs=crs)
        gdf_out.to_file(output_path, driver="GeoJSON")
    else:
        out_df.to_csv(output_path, index=False)
    logger.info(f"Wrote results to {output_path} ({args.output_format})")

    # Upload to S3 if requested
    if args.upload_s3:
        s3_key = args.s3_output_key
        if s3_key is None:
            s3_key = f"lidar_heights/{output_path.name}"
        logger.info(f"Uploading to s3://{args.s3_output_bucket}/{s3_key}")
        s3_client.upload_file(str(output_path), args.s3_output_bucket, s3_key)
        logger.info(f"Successfully uploaded to s3://{args.s3_output_bucket}/{s3_key}")


if __name__ == "__main__":
    main()
