"""
Pandora Pipeline - Cropping Module
Building facade extraction from rectified panorama images
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cv2
import logging

from .geodataframe_processor import GeoDataFrameProcessor

logger = logging.getLogger(__name__)


class BuildingProcessor:
    """Class for extracting buildings from rectified panorama images."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BuildingProcessor with configuration.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.crop_config = config.get('cropping', {})
        
        # Extract parameters
        self.building_offset = self.crop_config.get('building_offset', 0.8)
        self.image_width_rad = config.get('rectification', {}).get('fov_horizontal', 154) * np.pi / 180
        self.image_quality = self.crop_config.get('image_quality', 95)
        self.save_intermediate = self.crop_config.get('save_intermediate', False)
        self.min_building_width = self.crop_config.get('min_building_width', 30)  # Lowered from 50 to 30px
        self.min_building_height = self.crop_config.get('min_building_height', 100)
        
        logger.info(f"BuildingProcessor initialized with offset={self.building_offset}")
    
    @staticmethod
    def align_heading_map(heading_map: np.ndarray, 
                         heading_midpoint: float, 
                         heading_first: float, 
                         heading_second: float) -> np.ndarray:
        """
        Align heading map to match the expected heading values.
        
        Args:
            heading_map: 2D array of heading values
            heading_midpoint: Expected heading at building midpoint
            heading_first: Heading to first building edge
            heading_second: Heading to second building edge
            
        Returns:
            Normalized heading map with values in [-pi, pi]
        """
        # Find offset between heading map and expected values
        central_j = heading_map.shape[1] // 2
        central_heading = heading_map[-1, central_j]
        
        heading_offset = central_heading - heading_midpoint
        heading_map_normalized = heading_map - heading_offset

        # Handle wrap-around for angles [-pi, pi]
        heading_map_normalized = np.where(
            heading_map_normalized > np.pi, 
            heading_map_normalized - 2*np.pi, 
            heading_map_normalized
        )
        heading_map_normalized = np.where(
            heading_map_normalized < -np.pi, 
            heading_map_normalized + 2*np.pi, 
            heading_map_normalized
        )
        
        return heading_map_normalized
    
    def extract_building_facade(self, 
                                row: pd.Series, 
                                image_folder: str, 
                                save_crops: bool = True, 
                                output_dir: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Extract building facade from a rectified panorama image.
        
        Args:
            row: DataFrame row containing building information
            image_folder: Path to folder containing rectified images
            save_crops: Whether to save the cropped building images
            output_dir: Output directory for cropped images
            
        Returns:
            Cropped building image as numpy array, or None if extraction fails
        """
        try:
            # Extract image and heading information
            image_name = row["FOTO"].split(".")[0]
            json_name = f"{image_name}_heading_facade.json"
            json_path = os.path.join(image_folder, json_name)
            
            if not os.path.exists(json_path):
                logger.warning(f"Heading JSON not found: {json_path}")
                return None
            
            # Try to read JSON, handle corrupted files gracefully
            try:
                json_data = pd.read_json(json_path)
                if json_data.empty:
                    logger.warning(f"Heading JSON is empty: {json_path}")
                    return None
            except Exception as e:
                logger.warning(f"Failed to read heading JSON {json_path}: {e}")
                return None
            
            # Get building heading angles
            heading_midpoint = row["midpoint_yaw_rad"]
            heading_first = row["edge_first_yaw_rad"]
            heading_second = row["edge_second_yaw_rad"]

            logger.debug(f'Heading building midpoint: {heading_midpoint}')

            # Find match rectified image
            facade_orientation = row["orientation"]
            
            # Convert camera Yaw (degrees) to cardinal direction
            yaw_deg = row["Yaw"] if "Yaw" in row else row.get("orientation_run", 0.0)
            # Normalize to 0-360
            yaw_normalized = yaw_deg % 360
            # Convert to cardinal direction: N(0), E(90), S(180), W(270)
            # Using 45-degree ranges around each cardinal point
            if yaw_normalized < 45 or yaw_normalized >= 315:
                camera_orientation = "N"
            elif 45 <= yaw_normalized < 135:
                camera_orientation = "E"
            elif 135 <= yaw_normalized < 225:
                camera_orientation = "S"
            else:
                camera_orientation = "W"

            # Find if use 2 vanishing point (with fallback if 'i' column is missing)
            try:
                use_2_v_point = (json_data["i"] == 1).any() if "i" in json_data.columns else False
            except (KeyError, AttributeError):
                logger.debug(f"Column 'i' not found in heading JSON, assuming 1VP")
                use_2_v_point = False
            # Find rectified image side
            rectified_image_side = GeoDataFrameProcessor.calculate_cardinal_direction(
                camera_orientation, facade_orientation
            )

            if use_2_v_point:
                if json_data['heading'][0] < 0:
                    map_index_to_direction = {'right': 0, 'left': 1, 'forward': 0, 'backward': 1}
                else:
                    map_index_to_direction = {'right': 1, 'left': 0, 'forward': 1, 'backward': 0}
            else:
                if json_data['heading'][0] > 0:
                    map_index_to_direction = {'right': 1, 'left': 0, 'forward': 1, 'backward': 0}
                else:
                    map_index_to_direction = {'right': 0, 'left': 1, 'forward': 0, 'backward': 1}
            
            # Find best matching rectified image
            i = 0
            j = map_index_to_direction[rectified_image_side]

            logger.debug(f'Using VP indices: i={i}, j={j}')
            
            # Load rectified image and heading map (with fallback to other VP)
            rectified_image_path = os.path.join(image_folder, f"{image_name}_VP_{i}_{j}.jpg")
            heading_map_path = os.path.join(image_folder, f"{image_name}_VP_{i}_{j}_heading_map.npy")
            
            # Try primary VP, then fallback to other VP if not found
            if not os.path.exists(rectified_image_path) or not os.path.exists(heading_map_path):
                logger.debug(f"Primary VP {i}_{j} not found, trying fallback VP...")
                # Try all possible VP combinations: (0,0), (0,1), (1,0), (1,1)
                vp_candidates = [(0, 0), (0, 1), (1, 0), (1, 1)]
                found = False
                for vi, vj in vp_candidates:
                    alt_image_path = os.path.join(image_folder, f"{image_name}_VP_{vi}_{vj}.jpg")
                    alt_map_path = os.path.join(image_folder, f"{image_name}_VP_{vi}_{vj}_heading_map.npy")
                    if os.path.exists(alt_image_path) and os.path.exists(alt_map_path):
                        logger.debug(f"Using fallback VP {vi}_{vj}")
                        rectified_image_path = alt_image_path
                        heading_map_path = alt_map_path
                        i, j = vi, vj
                        found = True
                        break
                
                if not found:
                    logger.warning(f"Rectified image not found for all VP: {image_name}")
                    return None
            
            rectified_image = Image.open(rectified_image_path)
            heading_map = np.load(heading_map_path, allow_pickle=True)
            
            # Check if heading map is already compressed (1D = bottom row only)
            if heading_map.ndim == 1:
                logger.debug(f"Heading map is already bottom row only (shape: {heading_map.shape})")
                heading_map_bottom_row = heading_map
                # Create a 2D version for align function (single row)
                heading_map_2d = heading_map.reshape(1, -1)
                heading_map_aligned = self.align_heading_map(
                    heading_map_2d, heading_midpoint, heading_first, heading_second
                )
                heading_map_bottom_row = heading_map_aligned[0, :]
            else:
                # Full heading map - align and extract bottom row
                heading_map_aligned = self.align_heading_map(
                    heading_map, heading_midpoint, heading_first, heading_second
                )
                # Get bottom row of heading map (most accurate)
                heading_map_bottom_row = heading_map_aligned[-1, :]
            
            # Tile bottom row to match image height
            heading_map_tiled = np.tile(heading_map_bottom_row, (rectified_image.height, 1))
            
            logger.debug(f"Max heading: {heading_map_tiled.max():.3f}")
            logger.debug(f"Building edges: {heading_first:.3f}, {heading_second:.3f}")
            
            # Find column index closest to midpoint heading
            heading_diff = heading_map_tiled - heading_midpoint
            heading_midpoint_index = np.argmin(np.abs(heading_diff))
            logger.debug(f"Midpoint column index: {heading_midpoint_index}")

            # Fix issue on negative heading second
            if not heading_first < heading_midpoint < heading_second:
                if heading_second < 0:
                    heading_second = heading_midpoint + abs(heading_midpoint - heading_first)
            
            # Ensure correct ordering of edges
            if heading_first > heading_second:
                heading_first, heading_second = heading_second, heading_first
            
            # Calculate pixel indices for building edges
            heading_first_proportion = (heading_first - heading_midpoint) / self.image_width_rad
            heading_second_proportion = (heading_second - heading_midpoint) / self.image_width_rad
            
            logger.debug(f"Proportions: first={heading_first_proportion:.3f}, second={heading_second_proportion:.3f}")
            
            # Apply offsetting factor and calculate final indices
            heading_first_index = int(
                heading_midpoint_index + 
                self.building_offset * heading_first_proportion * heading_map_tiled.shape[1]
            )
            heading_second_index = int(
                heading_midpoint_index + 
                self.building_offset * heading_second_proportion * heading_map_tiled.shape[1]
            )
            
            # Ensure indices are within bounds BEFORE ordering
            heading_first_index = max(0, min(heading_map_tiled.shape[1], heading_first_index))
            heading_second_index = max(0, min(heading_map_tiled.shape[1], heading_second_index))
            
            # Ensure indices are correctly ordered (left < right)
            if heading_first_index > heading_second_index:
                heading_first_index, heading_second_index = heading_second_index, heading_first_index
            
            logger.debug(f"Building column indices: {heading_first_index} to {heading_second_index}")
            
            # Validate building dimensions
            building_width = heading_second_index - heading_first_index
            if building_width < self.min_building_width:
                logger.warning(f"Building too narrow: {building_width}px < {self.min_building_width}px")
                return None
            
            # Extract and save the cropped building
            if save_crops:
                cropped_building = self._extract_and_save_building(
                    rectified_image, 
                    heading_first_index, 
                    heading_second_index,
                    image_name,
                    output_dir or image_folder
                )
                return cropped_building
            else:
                # Extract without saving (still convert to BGR for consistency)
                img_array = np.array(rectified_image)
                img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                cropped_building = img_array_bgr[:, heading_first_index:heading_second_index]
                return cropped_building
                
        except Exception as e:
            logger.error(f"Error extracting building facade: {e}", exc_info=True)
            return None
    
    def _extract_and_save_building(self, 
                                   rectified_image: Image.Image, 
                                   first_col: int, 
                                   second_col: int,
                                   image_name: str,
                                   output_dir: str) -> np.ndarray:
        """
        Extract and save the cropped building region.
        
        Args:
            rectified_image: The rectified panorama image
            first_col: Starting column index
            second_col: Ending column index
            image_name: Base name of the image
            output_dir: Directory to save the cropped image
            
        Returns:
            The cropped building image as numpy array
        """
        # Convert PIL image to numpy array (PIL uses RGB)
        img_array = np.array(rectified_image)
        
        # Convert RGB to BGR for OpenCV compatibility
        img_array_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Extract the building region
        cropped_building = img_array_bgr[:, first_col:second_col]
        
        # Validate height
        if cropped_building.shape[0] < self.min_building_height:
            logger.warning(f"Building too short: {cropped_building.shape[0]}px < {self.min_building_height}px")
        
        # Create output directory for cropped buildings if it doesn't exist
        cropped_output_dir = os.path.join(output_dir, "cropped_buildings")
        os.makedirs(cropped_output_dir, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{image_name}_building_cropped.jpg"
        output_path = os.path.join(cropped_output_dir, output_filename)
        
        # Save using OpenCV with quality control (already in BGR format)
        cv2.imwrite(output_path, cropped_building, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
        
        logger.info(f"Saved cropped building: {output_path}")
        logger.debug(f"Cropped image size: {cropped_building.shape[1]}x{cropped_building.shape[0]} pixels")
        
        return cropped_building
    
    def visualize_extraction(self, 
                            rectified_image: Image.Image, 
                            first_col: int, 
                            second_col: int,
                            save_path: Optional[str] = None) -> None:
        """
        Create visualization of building extraction results.
        
        Args:
            rectified_image: The rectified panorama image
            first_col: Starting column index
            second_col: Ending column index
            save_path: Optional path to save visualization
        """
        # Create mask
        img_array = np.array(rectified_image)
        mask = np.ones((img_array.shape[0], img_array.shape[1]))
        mask[:, first_col:second_col] = 0
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original rectified image
        axes[0].imshow(rectified_image)
        axes[0].set_title("Rectified Image")
        axes[0].axis('off')
        
        # 2. Binary mask
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title("Building Region Mask")
        axes[1].axis('off')
        
        # 3. Overlay
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        axes[2].imshow(rectified_image)
        axes[2].imshow(mask, cmap=cmap)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization: {save_path}")
        else:
            plt.show()
        
        plt.close()
