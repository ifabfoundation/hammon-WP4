"""
Height extraction logic using cylindrical queries on LiDAR points.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def histogram_from_classes(classification: Iterable[int]) -> Counter:
    """Build a histogram of class ids."""
    return Counter(int(c) for c in classification)


def compute_height(
    midpoint: Tuple[float, float],
    points: Dict[str, np.ndarray],
    radii: Sequence[float] = (3.0, 5.0, 8.0),
    class_map: Optional[Dict[str, Iterable[int]]] = None,
    ground_percentile: float = 5.0,
    roof_percentile: float = 95.0,
    min_ground_points: int = 30,
    min_building_points: int = 30,
) -> Dict[str, object]:
    """
    Compute building height in a cylindrical neighborhood around a midpoint.

    Returns a dictionary with ground_z, roof_z, height_m, counts, flags, radius_used.
    """
    if class_map is None:
        class_map = {"ground": [2], "building": [6], "vegetation": [3, 4, 5]}

    x = points.get("x")
    y = points.get("y")
    z = points.get("z")
    cls = points.get("classification")

    if x is None or y is None or z is None or cls is None:
        raise ValueError("Points dictionary must include x, y, z, and classification arrays")

    result: Dict[str, object] = {
        "ground_z": None,
        "roof_z": None,
        "height_m": None,
        "radius_used": None,
        "flags": [],
        "counts": {},
    }

    ground_classes: Set[int] = set(int(c) for c in class_map.get("ground", []))
    building_classes: Set[int] = set(int(c) for c in class_map.get("building", []))

    mx, my = midpoint

    for radius in radii:
        r2 = radius * radius
        dist2 = (x - mx) ** 2 + (y - my) ** 2
        mask = dist2 <= r2
        if not mask.any():
            continue

        z_sel = z[mask]
        cls_sel = cls[mask]

        ground_mask = np.isin(cls_sel, list(ground_classes)) if ground_classes else np.ones_like(cls_sel, dtype=bool)
        bldg_mask = np.isin(cls_sel, list(building_classes)) if building_classes else np.ones_like(cls_sel, dtype=bool)

        ground_z_vals = z_sel[ground_mask]
        bldg_z_vals = z_sel[bldg_mask]

        # Fallbacks if not enough classed points
        if len(ground_z_vals) < min_ground_points:
            result["flags"].append(f"ground_sparse_r{radius}")
            ground_z_vals = z_sel

        if len(bldg_z_vals) < min_building_points:
            result["flags"].append(f"building_sparse_r{radius}")
            bldg_z_vals = z_sel

        ground_z = float(np.percentile(ground_z_vals, ground_percentile))
        roof_z = float(np.percentile(bldg_z_vals, roof_percentile))
        height = roof_z - ground_z

        result.update(
            {
                "ground_z": ground_z,
                "roof_z": roof_z,
                "height_m": height,
                "radius_used": radius,
                "counts": Counter(int(c) for c in cls_sel),
            }
        )

        if height <= 0:
            result["flags"].append("non_positive_height")
        return result

    result["flags"].append("no_points_in_radii")
    return result
