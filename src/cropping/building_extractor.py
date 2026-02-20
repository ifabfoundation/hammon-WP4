"""
Pandora Pipeline - Cropping Module
Main orchestrator for building facade extraction and sky removal
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import geopandas as gpd
import pandas as pd
import cv2

from .geodataframe_processor import GeoDataFrameProcessor
from .building_processor import BuildingProcessor
from .sky_cropper import SkyCropper

logger = logging.getLogger(__name__)


class BuildingExtractor:
    """
    Main class for extracting building facades from panoramic images.
    Orchestrates the full pipeline: building extraction + sky removal.
    """
    
    def __init__(self, config: Dict[str, Any], temp_dir: str, output_dir: str):
        """
        Initialize the building facade extractor.
        
        Args:
            config: Configuration dictionary from config.yaml
            temp_dir: Temporary directory for intermediate files
            output_dir: Output directory for final cropped images
        """
        self.config = config
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.crop_config = config.get('cropping', {})
        
        # Processing parameters
        self.sky_offset = self.crop_config.get('sky_detection', {}).get('sky_offset', 20)
        self.save_intermediate = self.crop_config.get('save_intermediate', False)
        self.image_quality = self.crop_config.get('image_quality', 95)
        
        # Create output directories
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize processors
        self.gdf_processor = GeoDataFrameProcessor()
        self.building_processor = BuildingProcessor(config)
        self.sky_cropper = SkyCropper(sky_offset=self.sky_offset)
        
        logger.info(f"BuildingExtractor initialized")
        logger.info(f"  Temp dir: {temp_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Sky offset: {self.sky_offset}px")

    def process_single_building(self, 
                                row: pd.Series, 
                                rectified_image_folder: str,
                                save_final: bool = True) -> Optional[str]:
        """
        Process a single building: extract facade and remove sky.
        
        Args:
            row: DataFrame row containing building information
            rectified_image_folder: Path to folder with rectified images
            save_final: Whether to save the final processed image
            
        Returns:
            Path to final processed image, or None if processing fails
        """
        try:
            image_name = row["FOTO"].split(".")[0]
            logger.info(f"Processing building from panorama: {image_name}")
            
            # Step 1: Extract building facade from rectified panorama
            cropped_building = self.building_processor.extract_building_facade(
                row=row,
                image_folder=rectified_image_folder,
                save_crops=self.save_intermediate,
                output_dir=self.temp_dir
            )
            
            if cropped_building is None:
                logger.warning(f"Failed to extract building facade for {image_name}")
                return None
            
            # Step 2: Remove sky from cropped building
            final_image = self.sky_cropper.process_building_image(
                image=cropped_building,
                output_dir=self.output_dir,
                filename=f"{image_name}_final.jpg",
                save_image=save_final,
                image_quality=self.image_quality
            )
            
            if save_final:
                final_path = os.path.join(self.output_dir, f"final_{image_name}_final.jpg")
                logger.info(f"✅ Successfully processed building: {final_path}")
                return final_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing building: {e}", exc_info=True)
            return None
    
    def process_from_geodataframe(self, 
                                  gdf_path: Path, 
                                  rectified_image_folder: str,
                                  limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all buildings from a GeoDataFrame/CSV file.
        
        Args:
            gdf_path: Path to GeoJSON or CSV file with building data
            rectified_image_folder: Path to folder with rectified images
            limit: Maximum number of buildings to process (None = all)
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Loading building data from: {gdf_path}")
        
        # Load data (support both GeoJSON and CSV)
        if str(gdf_path).endswith('.geojson'):
            gdf = self.gdf_processor.load_geojson(gdf_path)
        elif str(gdf_path).endswith('.csv'):
            gdf = self.gdf_processor.load_csv(gdf_path)
        else:
            raise ValueError(f"Unsupported file format: {gdf_path}")
        
        # Limit number of buildings if specified
        if limit is not None:
            gdf = gdf.head(limit)
            logger.info(f"Processing limited to first {limit} buildings")
        
        total = len(gdf)
        logger.info(f"Found {total} buildings to process")
        
        # Statistics
        stats = {
            'total': total,
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Process each building
        for idx, row in gdf.iterrows():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing building {idx + 1}/{total}")
                logger.info(f"{'='*60}")
                
                result = self.process_single_building(
                    row=row,
                    rectified_image_folder=rectified_image_folder,
                    save_final=True
                )
                
                if result is not None:
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error processing building at index {idx}: {e}")
                stats['failed'] += 1
                continue
        
        # Log final statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total buildings: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {stats['processed']/stats['total']*100:.1f}%")
        
        return stats
    
    def batch_process_from_s3(self, 
                             csv_key: str,
                             s3_client,
                             input_bucket: str,
                             batch_size: int = 100) -> Dict[str, Any]:
        """
        Process buildings in batches, downloading rectified images from S3.
        
        Args:
            csv_key: S3 key for the building CSV/GeoJSON
            s3_client: S3Client instance
            input_bucket: S3 bucket with rectified images
            batch_size: Number of buildings to process per batch
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Starting batch processing from S3")
        logger.info(f"  CSV key: {csv_key}")
        logger.info(f"  Bucket: {input_bucket}")
        logger.info(f"  Batch size: {batch_size}")
        
        # Download building data from S3
        logger.info(f"Downloading building data from S3...")
        csv_local_path = os.path.join(self.temp_dir, 'buildings.csv')
        s3_client.download_file(input_bucket, csv_key, csv_local_path)
        
        # Load building data
        df = pd.read_csv(csv_local_path)
        total = len(df)
        logger.info(f"Loaded {total} buildings from CSV")
        
        # Statistics
        stats = {
            'total': total,
            'processed': 0,
            'failed': 0,
            'batches': 0
        }
        
        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_df = df.iloc[batch_start:batch_end]
            
            stats['batches'] += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"BATCH {stats['batches']}: Buildings {batch_start+1} to {batch_end}")
            logger.info(f"{'='*60}")
            
            # Process each building in batch
            for idx, row in batch_df.iterrows():
                try:
                    # Download rectified images for this panorama from S3
                    panorama_name = row["FOTO"].split(".")[0]
                    
                    # Assuming rectified images are in S3 with pattern:
                    # rectification_results/{panorama_name}_VP_*.jpg
                    # This will be customized based on actual S3 structure
                    
                    result = self.process_single_building(
                        row=row,
                        rectified_image_folder=self.temp_dir,
                        save_final=True
                    )
                    
                    if result is not None:
                        stats['processed'] += 1
                        
                        # Upload final image to S3
                        final_key = f"crop_results/{panorama_name}_final.jpg"
                        s3_client.upload_file(result, input_bucket, final_key)
                        logger.info(f"Uploaded to S3: {final_key}")
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing building {idx}: {e}")
                    stats['failed'] += 1
                    continue
            
            logger.info(f"Batch {stats['batches']} complete: {stats['processed']} processed, {stats['failed']} failed")
        
        # Log final statistics
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total batches: {stats['batches']}")
        logger.info(f"Total buildings: {stats['total']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Success rate: {stats['processed']/stats['total']*100:.1f}%")
        
        return stats
