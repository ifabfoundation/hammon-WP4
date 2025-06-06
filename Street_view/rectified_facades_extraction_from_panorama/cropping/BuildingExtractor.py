import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import cv2
from typing import List, Tuple, Dict, Any, Optional

from cropping.geodaframe.GeoDataFrameProcessor import GeoDataFrameProcessor
from cropping.extractor.BuildingProcessor import BuildingProcessor
from cropping.extractor.CroppingSky import SkyCropper
from cropping.utils.Config import Config

class BuildingFacadeExtractor:
    """
    Main class for extracting building facades from panoramic images.
    Combines configuration and processing in a single class.
    """
    
    def __init__(self, 
                # Processing flags
                plot_redundant: bool = False,
                save_directly: bool = False,
                
                # Paths
                root_dir: str = 'Pano_new',
                country_city: str = 'New',
                images_base_folder: str = './Pano_new/New/images',
                geojson_path: str = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Desktop\GitHub\hammon-WP4\lsaa-dataset-master\Panorama_Rectification\Pano_new\New\Preprocessed_to_use_for_picture_extraction.geojson',
                
                # Processing parameters
                new_count: int = 5,
                offsetting: float = 0.8,
                image_width_rad: float = 2.68,
                
                # Output settings
                save_cropped_buildings: bool = True,
                cropped_image_quality: int = 95,
                output_dir: str = '.'):
        """
        Initialize the building facade extractor with direct configuration parameters.
        
        Args:
            plot_redundant: Whether to display redundant plots
            save_directly: Whether to save images directly
            root_dir: Main directory
            country_city: Subdirectory for country/city
            images_base_folder: Base folder for panorama images
            geojson_path: Path to GeoJSON file with building data
            new_count: Counter for new images
            offsetting: Scaling factor for building edge detection
            image_width_rad: Image width in radians
            save_cropped_buildings: Whether to save cropped building images
            cropped_image_quality: JPEG quality for saved crops (1-100)
        """
        # Initialize configuration with the provided parameters
        self.config = Config(
            plot_redundant=plot_redundant,
            save_directly=save_directly,
            root_dir=root_dir,
            country_city=country_city,
            images_base_folder=images_base_folder,
            geojson_path=geojson_path,
            new_count=new_count,
            offsetting=offsetting,
            image_width_rad=image_width_rad,
            save_cropped_buildings=save_cropped_buildings,
            cropped_image_quality=cropped_image_quality,
            output_dir=output_dir
        )
        
        # Create output directory if needed
        if not os.path.exists(self.config.rendering_output_folder):
            os.makedirs(self.config.rendering_output_folder)
        
        # Initialize GeoDataFrame processor
        self.gdf_processor = GeoDataFrameProcessor()
        
        # Initialize building extractor
        self.building_extractor = BuildingProcessor(self.config)
        
        # Initialize sky cropper for the final processing step
        self.sky_cropper = SkyCropper(sky_offset=20)

    def __add_yaw_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add yaw angle columns to GeoDataFrame."""
        processor = GeoDataFrameProcessor()
        
        # Ensure required columns exist
        gdf = processor.ensure_outer_edges(gdf)
        gdf = processor.ensure_midpoints(gdf)
        
        # Calculate yaw angles
        midpoint_yaws = []
        first_yaws = []
        second_yaws = []
        
        for idx, row in gdf.iterrows():
            cam_x = row["X"]
            cam_y = row["Y"]
            
            # Midpoint yaw
            midpoint = row["geometry_midpoint"]
            midpoint_yaws.append(processor.yaw_radians(cam_x, cam_y, midpoint.x, midpoint.y))
            
            # Edge vertices yaws
            edge = row["geometry_outer_edge_shapely"]
            v1, v2 = processor.first_two_vertices(edge)
            first_yaws.append(processor.yaw_radians(cam_x, cam_y, v1.x, v1.y))
            
            if v2 is not None:
                second_yaws.append(processor.yaw_radians(cam_x, cam_y, v2.x, v2.y))
            else:
                second_yaws.append(None)
        
        # Add columns to DataFrame
        gdf["midpoint_yaw_rad"] = midpoint_yaws
        gdf["edge_first_yaw_rad"] = first_yaws
        gdf["edge_second_yaw_rad"] = second_yaws
        
        return gdf

    def extract_all_buildings(self, image_folder: Optional[str] = None) -> None:
        """Extract all buildings from the GeoDataFrame."""
        
        # Load and process GeoDataFrame
        processor = GeoDataFrameProcessor()
        gdf_combined = processor.load_geojson(Path(self.config.GEOJSON_PATH))
        
        # Add yaw angle columns
        gdf_combined = self.__add_yaw_columns(gdf_combined)
        
        # Initialize building extractor
        extractor = BuildingProcessor(self.config)
        
        # Process each building
        for idx, row in gdf_combined.iterrows():
            try:
                print(f"\n{'='*60}")
                print(f"Processing building {idx + 1}/{len(gdf_combined)}")
                print(f"{'='*60}")
                
                # Extract building facade
                building_result = extractor.extract_building_facade(row, self.config.image_data_folder, 
                                                                  save_crops=True, 
                                                                  output_dir=self.config.rendering_output_folder)
                
                # Get the basename of the image for the final output filename
                image_name = row["FOTO"].split(".")[0]
                filename = f"{image_name}_building_cropped.jpg"
                
                # Final processing step: remove sky from the extracted building image
                # Look for the cropped image file
                cropped_file_path = os.path.join(self.config.rendering_output_folder, filename)
                
                if os.path.exists(cropped_file_path):
                    # Create a directory for the final images
                    final_output_dir = os.path.join(self.config.rendering_output_folder, "final_images")
                    
                    # Load the cropped image
                    cropped_image = cv2.imread(cropped_file_path)
                    
                    if cropped_image is not None:
                        # Apply sky cropping
                        self.sky_cropper.process_building_image(cropped_image, final_output_dir, filename)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
