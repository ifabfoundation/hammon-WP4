from cropping.utils.libraries import *
from cropping.geodaframe.GeoDataFrameProcessor import GeoDataFrameProcessor
from cropping.utils.Config import Config

# Building Extraction Functions
class BuildingProcessor:
    """Class for extracting buildings from rectified panorama images."""
    
    def __init__(self, config: Config):
        self.config = config
    
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
    
    def extract_building_facade(self, row: pd.Series, image_folder: str, save_crops: bool = True, output_dir: Optional[str] = None) -> None:
        """
        Extract building facade from a rectified panorama image.
        
        Args:
            row: DataFrame row containing building information
            image_folder: Path to folder containing rectified images
            save_crops: Whether to save the cropped building images
        """
        # Extract image and heading information
        image_name = row["FOTO"].split(".")[0]
        json_name = f"{image_name}_heading_facade.json"
        json_data = pd.read_json(os.path.join(image_folder, json_name))
        
        # Get building heading angles
        heading_midpoint = row["midpoint_yaw_rad"]
        heading_first = row["edge_first_yaw_rad"]
        heading_second = row["edge_second_yaw_rad"]

        print('Heading building: ', heading_midpoint)

        # Find match rectified image
        facade_orientation = row["orientation"]
        camera_orientation = row["orientation_run"]

        # Find if use 2 vanishing point
        use_2_v_point = (json_data["i"] == 1).any()
        # Find rectified image side
        rectified_image_side = GeoDataFrameProcessor.calculate_cardinal_direction(camera_orientation, facade_orientation)

        if use_2_v_point:
            if json_data['heading'][0] < 0:
                map_index_to_direction = {
                    'right': 0,
                    'left': 1,
                }
            else:
                map_index_to_direction = {
                    'right': 1,
                    'left': 0,
                }
        else:
            if json_data['heading'][0] > 0:
                map_index_to_direction = {
                    'right': 1,
                    'left': 0,
                }
            else:
                map_index_to_direction = {
                    'right': 0,
                    'left': 1,
                }
        
        # Find best matching rectified image
        i = 0
        j = map_index_to_direction[rectified_image_side]

        print('i: ', i)
        print('j: ', j)
        
        print(json_data)
        
        # Load rectified image and heading map
        rectified_image_path = f"{image_name}_VP_{i}_{j}.jpg"
        heading_map_path = f"{image_name}_VP_{i}_{j}_heading_map.npy"
        
        rectified_image = Image.open(os.path.join(image_folder, rectified_image_path))
        heading_map = np.load(os.path.join(image_folder, heading_map_path), allow_pickle=True)
        
        # Align heading map
        heading_map_aligned = self.align_heading_map(
            heading_map, heading_midpoint, heading_first, heading_second
        )
        
        # Get bottom row of heading map (most accurate)
        heading_map_bottom_row = heading_map_aligned[-1, :]
        heading_map_tiled = np.tile(heading_map_bottom_row, (heading_map_aligned.shape[0], 1))
        
        print(f"Max heading: {heading_map_tiled.max()}")
        print(f"Building edges: {heading_first}, {heading_second}")
        
        # Visualization of heading distribution
        # plt.figure()
        # plt.scatter(np.arange(heading_map_aligned.shape[1]), heading_map_bottom_row)
        # plt.title("Bottom Row Heading Distribution")
        # plt.xlabel("Column Index")
        # plt.ylabel("Heading (radians)")
        
        # Create initial mask using logical conditions
        mask_logical = np.logical_and(
            heading_map_aligned > heading_first, 
            heading_map_aligned < heading_second
        )
        heading_map_aligned[mask_logical] = 0
        
        # Find column index closest to midpoint heading
        heading_diff = heading_map_tiled - heading_midpoint
        heading_midpoint_index = np.argmin(np.abs(heading_diff))
        print(f"Midpoint column index: {heading_midpoint_index}")

        # Fix issue on negative heading second
        if not heading_first < heading_midpoint < heading_second:
            if heading_second < 0:
                heading_second = heading_midpoint + abs(heading_midpoint - heading_first)
        
        # Ensure correct ordering of edges
        if heading_first > heading_second:
            heading_first, heading_second = heading_second, heading_first
        
        # Calculate pixel indices for building edges
        heading_first_proportion = (heading_first - heading_midpoint) / self.config.IMAGE_WIDTH_RAD
        heading_second_proportion = (heading_second - heading_midpoint) / self.config.IMAGE_WIDTH_RAD
        
        print(f"Proportions: first={heading_first_proportion:.3f}, second={heading_second_proportion:.3f}")
        
        # Apply offsetting factor and calculate final indices
        heading_first_index = int(
            heading_midpoint_index + 
            self.config.OFFSETTING * heading_first_proportion * heading_map_tiled.shape[1]
        )
        heading_second_index = int(
            heading_midpoint_index + 
            self.config.OFFSETTING * heading_second_proportion * heading_map_tiled.shape[1]
        )
        
        print(f"Building column indices: {heading_first_index} to {heading_second_index}")
        
        # Create final mask
        mask = np.ones(heading_map_tiled.shape)
        mask[:, heading_first_index:heading_second_index] = 0
        
        # Extract and save the cropped building
        if save_crops:
            cropped_building = self._extract_and_save_building(
                rectified_image, 
                heading_first_index, 
                heading_second_index,
                image_name,
                image_folder,
                output_dir
            )
        
        # Visualization
        self._visualize_extraction(rectified_image, mask)
    
    def _extract_and_save_building(self, 
                                   rectified_image: Image.Image, 
                                   first_col: int, 
                                   second_col: int,
                                   image_name: str,
                                   image_folder: str,
                                   output_dir: Optional[str] = None) -> np.ndarray:
        """
        Extract and save the cropped building region.
        
        Args:
            rectified_image: The rectified panorama image
            first_col: Starting column index
            second_col: Ending column index
            image_name: Base name of the image
            image_folder: Folder to save the cropped image
            
        Returns:
            The cropped building image as numpy array
        """
        # Convert PIL image to numpy array
        img_array = np.array(rectified_image)
        
        # Extract the building region
        cropped_building = img_array[:, first_col:second_col]
        
        # Create output directory for cropped buildings if it doesn't exist
        if output_dir is None:
            output_dir = os.path.join(image_folder, "cropped_buildings")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate output filename
        output_filename = f"{image_name}_building_cropped.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the cropped image
        cropped_pil = Image.fromarray(cropped_building)
        cropped_pil.save(output_path, quality=95)
        
        print(f"Saved cropped building to: {output_path}")
        print(f"Cropped image size: {cropped_building.shape[1]}x{cropped_building.shape[0]} pixels")
        
        return cropped_building
    
    def _visualize_extraction(self, rectified_image: Image.Image, mask: np.ndarray) -> None:
        """Create visualization of building extraction results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original rectified image
        axes[0].imshow(rectified_image)
        axes[0].set_title("Immagine Rettificata")
        axes[0].axis('off')
        
        # 2. Binary mask
        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title("Regioni con heading 88 - 10 gradi")
        axes[1].axis('off')
        
        # 3. Overlay
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        axes[2].imshow(rectified_image)
        axes[2].imshow(mask, cmap=cmap)
        axes[2].set_title("Sovrapposizione")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
