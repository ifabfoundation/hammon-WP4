from cropping.utils.libraries import *
from s3_library.S3Client import S3Client

class Config:
    """
    Configuration parameters for the panorama processing pipeline.
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
                 offsetting: float = 0.8,  # Scaling factor for building edge detection
                 image_width_rad: float = 2.68,  # Image width in radians (154 degrees)
                 
                 # Output settings
                 save_cropped_buildings: bool = True,  # Whether to save cropped building images
                 cropped_image_quality: int = 95,  # JPEG quality for saved crops (1-100)
                 output_dir: Optional[str] = None,
                 
                 # S3 settings
                 s3_client: Optional[S3Client] = None,
                ):
        """
        Initialize configuration with customizable parameters.
        
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
        # Processing flags
        self.PLOT_REDUNDANT = plot_redundant
        self.SAVE_DIRECTLY = save_directly
        
        # Paths
        self.ROOT_DIR = root_dir
        self.COUNTRY_CITY = country_city
        self.GEOJSON_PATH = geojson_path
        self.ZONE_NAME = zone_name   

        # Processing parameters
        self.NEW_COUNT = new_count
        self.OFFSETTING = offsetting
        self.IMAGE_WIDTH_RAD = image_width_rad
        
        # Output settings
        self.SAVE_CROPPED_BUILDINGS = save_cropped_buildings
        self.CROPPED_IMAGE_QUALITY = cropped_image_quality
        self.OUTPUT_DIR = output_dir
        
        # S3 settings
        if s3_client is not None:
            self.s3_client = s3_client
        else:
            self.s3_client = S3Client()
        
        # Initialize derived paths
        #self.tmp_count = str(self.NEW_COUNT)
        #self.inter_dir = os.path.join(self.ROOT_DIR, 'Pano_hl_z_vp/')
        self.rendering_output_folder = os.path.join(self.OUTPUT_DIR, self.ZONE_NAME, 'cropped_buildings/')
        self.image_data_folder = os.path.join(self.ROOT_DIR, self.COUNTRY_CITY, self.ZONE_NAME, 'Rendering/')
        #self.output_geojson = Path(self.GEOJSON_PATH).with_name(
        #    Path(self.GEOJSON_PATH).stem + "_with_yaw.geojson"
        #)
        
