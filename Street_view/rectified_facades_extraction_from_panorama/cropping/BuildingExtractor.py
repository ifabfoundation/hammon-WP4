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

from s3_library.S3Client import S3Client

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
                geojson_path: str = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Desktop\GitHub\hammon-WP4\lsaa-dataset-master\Panorama_Rectification\Pano_new\New\Preprocessed_to_use_for_picture_extraction.geojson',
                zone_name: str = 'default',
                
                # Processing parameters
                new_count: int = 5,
                offsetting: float = 0.8,
                image_width_rad: float = 2.68,
                
                # Output settings
                save_cropped_buildings: bool = True,
                cropped_image_quality: int = 95,
                output_dir: str = '.',
                
                # S3 settings
                s3_client: S3Client = None):
        """
        Initialize the building facade extractor with direct configuration parameters.
        
        Args:
            plot_redundant: Whether to display redundant plots
            save_directly: Whether to save images directly
            root_dir: Main directory
            country_city: Subdirectory for country/city
            geojson_path: Path to GeoJSON file with building data
            zone_name: Name of the zone
            new_count: Counter for new images
            offsetting: Scaling factor for building edge detection
            image_width_rad: Image width in radians
            save_cropped_buildings: Whether to save cropped building images
            cropped_image_quality: JPEG quality for saved crops (1-100)
            output_dir: Directory to save cropped buildings
            s3_client: S3 client for uploading files
        """
        # Initialize configuration with the provided parameters
        self.config = Config(
            plot_redundant=plot_redundant,
            save_directly=save_directly,
            root_dir=root_dir,
            country_city=country_city,
            geojson_path=geojson_path,
            zone_name=zone_name,
            new_count=new_count,
            offsetting=offsetting,
            image_width_rad=image_width_rad,
            save_cropped_buildings=save_cropped_buildings,
            cropped_image_quality=cropped_image_quality,
            output_dir=output_dir,
            s3_client=s3_client
        )
        
        # Initialize GeoDataFrame processor
        self.gdf_processor = GeoDataFrameProcessor(self.config.s3_client)
        
        # Initialize building extractor
        self.building_extractor = BuildingProcessor(self.config)
        
        # Initialize sky cropper for the final processing step
        self.sky_cropper = SkyCropper(self.config, sky_offset=20)

    def extract_all_buildings(self, image_folder: Optional[str] = None) -> None:
        """Extract all buildings from the GeoDataFrame."""
        
        # Load and process GeoDataFrame
        gdf_combined = self.gdf_processor.load_geojson(self.config.GEOJSON_PATH)
        
        # Add yaw angle columns
        #gdf_combined = self.__add_yaw_columns(gdf_combined)
        
        # Process each building
        for idx, row in gdf_combined.iterrows():
            try:
                print(f"\n{'='*60}")
                print(f"Processing building {idx + 1}/{len(gdf_combined)}")
                print(f"{'='*60}")
                
                # Extract building facade
                building_result = self.building_extractor.extract_building_facade(row, self.config.image_data_folder, 
                                                                  save_crops=True, 
                                                                  output_dir=self.config.rendering_output_folder)
                
                # Get the basename of the image for the final output filename
                image_name = row["FOTO"].split(".")[0]
                filename = f"{image_name}_building_cropped.jpg"
                
                # Final processing step: remove sky from the extracted building image
                # Look for the cropped image file
                cropped_file_path = os.path.join(self.config.rendering_output_folder, filename)
                cropped_files = self.config.s3_client.list_files('data', cropped_file_path)
                
                if 'Contents' in cropped_files:
                    # Create a directory for the final images
                    final_output_dir = os.path.join(self.config.rendering_output_folder, "final_images")
                    
                    # Load the cropped image
                    cropped_image = self.config.s3_client.read_cv2_image('data', cropped_file_path)
                    
                    if cropped_image is not None:
                        # Apply sky cropping
                        self.sky_cropper.process_building_image(cropped_image, final_output_dir, filename)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
