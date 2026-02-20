"""
Pandora Pipeline - Panorama Processor
Rectifies panoramic images using Simon's algorithm with vanishing points detection.
Adapted from old_sw/rectification/classes/PanoramaProcessor.py
"""

import os
import glob
import logging
import numpy as np
from PIL import Image
import skimage.io
from typing import Dict, Any, Optional, Tuple, List

# Import internal rectification modules
from .util.libraries import (
    simon_rectification,
    project_facade_for_refine,
    render_imgs,
    calculate_consensus_zp,
    calculate_histogram,
    R_heading,
    R_roll,
    R_pitch,
    draw_all_vp_and_hl_color,
    draw_all_vp_and_hl_bi,
    draw_sphere_zenith,
    draw_consensus_zp_hvps,
    draw_consensus_rectified_sphere,
    draw_center_hvps_rectified_sphere,
    draw_center_hvps_on_panorams
)
from .vanishing_points_utils import Pano_hvp

logger = logging.getLogger(__name__)


class PanoramaProcessor:
    """
    Process panoramic images and extract rectified facades using Simon's algorithm.
    
    This class handles:
    1. Tile generation from panoramic images
    2. Vanishing points detection on each tile
    3. Consensus calculation for zenith and horizontal VPs
    4. Image rectification and facade extraction
    5. Heading map generation (with optimization for bottom row only)
    """
    
    def __init__(self, config: Dict[str, Any], temp_dir: str, output_dir: str):
        """
        Initialize the panorama processor.
        
        Args:
            config: Configuration dictionary from config.yaml
            temp_dir: Temporary directory for intermediate files
            output_dir: Output directory for rectified images
        """
        self.config = config
        self.temp_dir = temp_dir
        self.output_dir = output_dir
        self.rect_config = config.get('rectification', {})
        
        # Processing parameters
        self.plot_redundant = self.rect_config.get('plot_redundant', False)
        self.save_directly = self.rect_config.get('save_directly', True)
        self.save_heading_map = self.rect_config.get('save_heading_map', 'bottom_row_only')
        self.image_quality = self.rect_config.get('image_quality', 95)
        
        # Create output directories
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"PanoramaProcessor initialized")
        logger.info(f"  Temp dir: {temp_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Heading map mode: {self.save_heading_map}")
    
    def setup_temp_folders(self, panorama_name: str) -> Tuple[str, str]:
        """
        Configure temporary folders for a specific panorama.
        
        Args:
            panorama_name: Name of the panorama (without extension)
            
        Returns:
            Tuple of (tmp_folder, tmp_folder_ifab)
        """
        tmp_folder = os.path.join(self.temp_dir, panorama_name, 'tiles')
        tmp_folder_ifab = os.path.join(self.temp_dir, panorama_name, 'ifab')
        
        os.makedirs(tmp_folder, exist_ok=True)
        os.makedirs(tmp_folder_ifab, exist_ok=True)
        
        # Remove old temporary files
        for pattern in ['*.jpg', '*.npy', '*.json']:
            for file_path in glob.glob(os.path.join(tmp_folder, pattern)):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
        
        return tmp_folder, tmp_folder_ifab
    
    def process_tiles(self, panorama_img: np.ndarray, tmp_folder: str, 
                     tmp_folder_ifab: str, inter_dir: str, root_dir: str, 
                     new_count: Optional[float] = None) -> Tuple:
        """
        Generate and process tiles from the panoramic image.
        
        Args:
            panorama_img: Panoramic image as numpy array
            tmp_folder: Temporary folder for tiles
            tmp_folder_ifab: IFAB temporary folder
            inter_dir: Intermediate directory
            root_dir: Root directory
            new_count: LSD scale parameter (default: from config or 0.8)
            
        Returns:
            Tuple containing: (tilelist, hl, hvps, hvp_groups, z, z_group, ls,
                              z_homo, hvp_homo, ls_homo, params)
        """
        # Get LSD scale from config or use default
        if new_count is None:
            vp_config = self.rect_config.get('vp_detection', {})
            new_count = vp_config.get('lsd_scale', 0.8)
        
        logger.info(f"Generating tiles from panorama (LSD scale: {new_count})...")
        
        # Generate tiles from the panoramic image
        tilelist = render_imgs(panorama_img, tmp_folder, tmp_folder_ifab, self.save_directly)
        
        if not self.save_directly:
            # Use already saved images
            tilelist = glob.glob(os.path.join(tmp_folder, '*.jpg'))
            tilelist.sort()
        
        logger.info(f"Generated {len(tilelist)} tiles")
        
        # Initialize lists for rectification data
        hl = []
        hvps = []
        hvp_groups = []
        z = []
        z_group = []
        ls = []
        z_homo = []
        hvp_homo = []
        ls_homo = []
        params = None
        
        # Process each tile
        for i, tile_path in enumerate(tilelist):
            logger.debug(f"Processing tile {i+1}/{len(tilelist)}: {tile_path}")
            
            [tmp_hl, tmp_hvps, tmp_hvp_groups, tmp_z, tmp_z_group, tmp_ls,
             tmp_z_homo, tmp_hvp_homo, tmp_ls_homo, params] = simon_rectification(
                tile_path, i, inter_dir, root_dir, new_count)
            
            hl.append(tmp_hl)
            hvps.append(tmp_hvps)
            hvp_groups.append(tmp_hvp_groups)
            z.append(tmp_z)
            z_group.append(tmp_z_group)
            ls.append(tmp_ls)
            z_homo.append(tmp_z_homo)
            hvp_homo.append(tmp_hvp_homo)
            ls_homo.append(tmp_ls_homo)
        
        logger.info(f"Tiles processing completed")
        return tilelist, hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, ls_homo, params
    
    def calculate_zenith_points(self, z_homo: List, hvp_homo: List, 
                               im: Image.Image, root_dir: str) -> Tuple[np.ndarray, List]:
        """
        Calculate zenith points from all perspectives.
        
        Args:
            z_homo: Zenith points in homogeneous coordinates
            hvp_homo: Horizontal vanishing points in homogeneous coordinates
            im: Panoramic image
            root_dir: Root directory for plots
            
        Returns:
            Tuple of (zenith_points, hv_points)
        """
        logger.info("Calculating zenith points from all perspectives...")
        
        # Calculate zenith points from all perspectives
        zenith_points = np.array([R_heading(np.pi / 2 * (i - 1)).dot(zenith) 
                                 for i, zenith in enumerate(z_homo)])
        points2 = np.array([R_heading(np.pi / 2 * (i - 1)).dot(np.array([0., 0., 1.])) 
                           for i in range(len(z_homo))])
        hv_points = [(R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T 
                    for i, hv_p in enumerate(hvp_homo)]
        
        if self.plot_redundant:
            logger.debug("Generating debug visualizations...")
            draw_all_vp_and_hl_color(zenith_points, hv_points, im.copy(), root_dir)
            draw_all_vp_and_hl_bi(zenith_points, hv_points, im.copy(), root_dir)
            draw_sphere_zenith(zenith_points, hv_points, root_dir)
        
        return zenith_points, hv_points
    
    def calculate_consensus(self, zenith_points: np.ndarray, ls_homo: List, 
                           im: Image.Image, params: Any, root_dir: str) -> Tuple:
        """
        Calculate the consensus zenith and horizontal vanishing points.
        
        Args:
            zenith_points: Zenith points from all tiles
            ls_homo: Lines in homogeneous coordinates
            im: Panoramic image
            params: Processing parameters
            root_dir: Root directory for plots
            
        Returns:
            Tuple of (best_zenith, final_hvps_rectified, pitch, roll)
        """
        logger.info("Calculating consensus zenith and horizontal vanishing points...")
        
        # Calculate the consensus zenith using SVD method
        [zenith_consensus, best_zenith] = calculate_consensus_zp(zenith_points, method='svd')
        
        logger.debug(f"Best zenith: {best_zenith}")
        
        # Transform consensus zenith points to original coordinates
        zenith_consensus_org = np.array([R_heading(-np.pi / 2 * (i - 1)).dot(zenith)
                                        for i, zenith in enumerate(zenith_consensus)])
        
        # Calculate horizontal vanishing points from the consensus zenith point
        result_list = []
        for i in range(len(zenith_consensus_org)):
            result = Pano_hvp.get_all_hvps(ls_homo[i], zenith_consensus_org[i], params)
            result_list.append(result)
        
        hvps_consensus_org = []
        for i in range(len(result_list)):
            hvps_consensus_org.append(result_list[i])
        
        hvps_consensus_uni = [(R_heading(np.pi / 2 * (i - 1)).dot(hv_p.T)).T
                             for i, hv_p in enumerate(hvps_consensus_org)]
        
        if self.plot_redundant:
            draw_consensus_zp_hvps(best_zenith, hvps_consensus_uni, im.copy(), root_dir)
        
        # Calculate pitch and roll from the best zenith
        pitch = np.arctan(best_zenith[2] / best_zenith[1])
        roll = -np.arctan(best_zenith[0] / np.sign(best_zenith[1]) *
                         np.hypot(best_zenith[1], best_zenith[2]))
        
        logger.debug(f"Pitch: {np.degrees(pitch):.2f}°, Roll: {np.degrees(roll):.2f}°")
        
        # Rectify horizontal vanishing points
        hvps_consensus_rectified = [R_roll(-roll).dot(R_pitch(-pitch).dot(vp.T)).T
                                   for vp in hvps_consensus_uni]
        
        if self.plot_redundant:
            draw_consensus_rectified_sphere(hvps_consensus_rectified, root_dir)
        
        # Calculate histogram of horizontal vanishing points to find dominant directions
        final_hvps_rectified = calculate_histogram(hvps_consensus_rectified, root_dir, 
                                                   self.plot_redundant)
        
        if self.plot_redundant:
            draw_center_hvps_rectified_sphere(np.array(final_hvps_rectified), root_dir)
            draw_center_hvps_on_panorams(best_zenith, np.array(final_hvps_rectified),
                                        im.copy(), pitch, roll, root_dir)
        
        return best_zenith, final_hvps_rectified, pitch, roll
    
    def process_single_panorama(self, panorama_path: str, panorama_name: str) -> Dict[str, Any]:
        """
        Process a single panoramic image through the complete rectification pipeline.
        
        Args:
            panorama_path: Path to the panoramic image file
            panorama_name: Name of the panorama (without extension)
            
        Returns:
            Dictionary with processing results and metadata
        """
        logger.info(f"Processing panorama: {panorama_name}")
        
        try:
            # Load the panoramic image
            im = Image.open(panorama_path)
            panorama_img = skimage.io.imread(panorama_path)
            
            # Setup temporary folders
            tmp_folder, tmp_folder_ifab = self.setup_temp_folders(panorama_name)
            
            # Create inter_dir and root_dir for internal functions
            inter_dir = os.path.join(self.temp_dir, panorama_name, 'inter')
            root_dir = self.temp_dir
            os.makedirs(inter_dir, exist_ok=True)
            
            # Process the tiles to detect vanishing points
            (tilelist, hl, hvps, hvp_groups, z, z_group, ls, z_homo, hvp_homo, 
             ls_homo, params) = self.process_tiles(
                panorama_img, tmp_folder, tmp_folder_ifab, inter_dir, root_dir)
            
            # Calculate zenith points from all tiles
            zenith_points, hv_points = self.calculate_zenith_points(
                z_homo, hvp_homo, im, root_dir)
            
            # Calculate consensus and get pitch/roll
            best_zenith, final_hvps_rectified, pitch, roll = self.calculate_consensus(
                zenith_points, ls_homo, im, params, root_dir)
            
            # Create output base path
            output_base = os.path.join(self.output_dir, panorama_name)
            
            # Get LSD scale for refine step
            vp_config = self.rect_config.get('vp_detection', {})
            lsd_scale = vp_config.get('lsd_scale', 0.8)
            
            # Render rectified facades with heading map generation
            logger.info("Rendering rectified facades...")
            project_facade_for_refine(
                np.array(final_hvps_rectified), 
                im.copy(), 
                pitch, 
                roll,
                panorama_path, 
                root_dir, 
                tmp_folder, 
                output_base, 
                lsd_scale,  # Use configured LSD scale instead of 0
                save_heading_map=self.save_heading_map  # Pass the optimization flag
            )
            
            # Clean up temporary tiles
            for file_path in glob.glob(os.path.join(tmp_folder, '*.jpg')):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {file_path}: {e}")
            
            logger.info(f"Panorama {panorama_name} processed successfully")
            
            # Return metadata
            return {
                'panorama_name': panorama_name,
                'status': 'success',
                'pitch': float(pitch),
                'roll': float(roll),
                'zenith': best_zenith.tolist() if isinstance(best_zenith, np.ndarray) else best_zenith,
                'num_tiles': len(tilelist),
                'num_hvps': len(final_hvps_rectified),
                'output_base': output_base
            }
            
        except Exception as e:
            logger.error(f"Error processing panorama {panorama_name}: {str(e)}", exc_info=True)
            return {
                'panorama_name': panorama_name,
                'status': 'failed',
                'error': str(e)
            }
