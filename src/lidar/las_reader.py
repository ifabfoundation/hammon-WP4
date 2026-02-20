"""
Utilities for listing, downloading, and reading LAS files for LiDAR height extraction.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np

from src.common.s3_client import S3Client

logger = logging.getLogger(__name__)


class LasRunLoader:
    """Helper to fetch LAS files for a run and expose basic summaries."""

    def __init__(
        self,
        s3_client: S3Client,
        bucket: str,
        prefix: str,
        temp_dir: str = "tmp/lidar",
    ):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _run_prefix(self, zone: str, run: str | int) -> str:
        return f"{self.prefix}/{zone}/laser/{run}/"

    def list_run_files(self, zone: str, run: str | int) -> List[str]:
        """List LAS keys for a given zone/run on S3."""
        prefix = self._run_prefix(zone, run)
        keys = [
            key
            for key in self.s3_client.list_files(self.bucket, prefix)
            if key.lower().endswith(".las")
        ]
        logger.info(f"Found {len(keys)} LAS files under {prefix}")
        return keys

    def download_run_files(
        self,
        zone: str,
        run: str | int,
        keys: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """Download all LAS files for a run; return local paths."""
        if keys is None:
            keys = self.list_run_files(zone, run)

        local_paths: List[Path] = []
        for key in keys:
            filename = Path(key).name
            local_path = self.temp_dir / str(zone) / str(run) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_path.exists():
                logger.info(f"Downloading s3://{self.bucket}/{key} -> {local_path}")
                self.s3_client.download_file(self.bucket, key, str(local_path))
            else:
                logger.debug(f"Using cached LAS: {local_path}")

            local_paths.append(local_path)

        return local_paths

    @staticmethod
    def summarize_classes(local_paths: Iterable[Path], chunk_size: int = 5_000_000) -> Counter:
        """Return a histogram of classification values across given LAS files."""
        counts: Counter = Counter()
        for path in local_paths:
            with laspy.open(path) as f:
                for chunk in f.chunk_iterator(chunk_size):
                    counts.update(chunk.classification.tolist())
        return counts

    @staticmethod
    def summarize_header(path: Path) -> Dict[str, object]:
        """Extract basic header info."""
        with laspy.open(path) as f:
            hdr = f.header
            return {
                "point_count": hdr.point_count,
                "scale": hdr.scales,
                "offset": hdr.offsets,
                "crs": hdr.parse_crs(),
                "point_format": hdr.point_format,
            }

    @staticmethod
    def load_points(
        local_paths: Iterable[Path],
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load points from LAS files into numpy arrays.

        Args:
            local_paths: LAS files to read.
            bbox: Optional (minx, miny, maxx, maxy) filter before concatenation.

        Returns:
            Dict with x, y, z, classification arrays concatenated across files.
        """
        x_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        z_list: List[np.ndarray] = []
        cls_list: List[np.ndarray] = []

        for path in local_paths:
            las = laspy.read(path)
            x = np.asarray(las.x)
            y = np.asarray(las.y)
            z = np.asarray(las.z)
            cls = np.asarray(las.classification)

            if bbox is not None:
                minx, miny, maxx, maxy = bbox
                mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
                x, y, z, cls = x[mask], y[mask], z[mask], cls[mask]

            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
            cls_list.append(cls)

        if not x_list:
            return {"x": np.array([]), "y": np.array([]), "z": np.array([]), "classification": np.array([])}

        return {
            "x": np.concatenate(x_list),
            "y": np.concatenate(y_list),
            "z": np.concatenate(z_list),
            "classification": np.concatenate(cls_list),
        }
