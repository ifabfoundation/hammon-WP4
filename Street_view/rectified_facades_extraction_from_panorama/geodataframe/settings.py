# %%
# Configuration parameters
CAMERA_FILE_PATH = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Documents - Hammon\WP4\data\DEMO_TRACCIATO/RUN.shp'
PANO_FOLDER_PATH = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Documents - Hammon\WP4\data\PANO_new'
BUFFER_METERS = 200
TARGET_CRS = "EPSG:7791"
OUTPUT_DIR = r'C:\Users\RaimondoReggio\OneDrive - Net Service S.p.A\Desktop\GitHub\hammon-WP4\Street_view\rectified_facades_extraction_from_panorama\geodataframe\results'
INTERACTIVE_PLOTS = True
LINE_OF_SIGHT_RADIUS = 100
BUILDING_DENSITY_THRESHOLD = 5

# Building area filtering parameters
MIN_BUILDING_AREA = 20
AREA_PERCENTILE_THRESHOLD = 5

# Enhanced line-of-sight parameters
DEFAULT_BUILDING_HEIGHT = 10.0
DEFAULT_FACADE_HEIGHT = 3.0
DEFAULT_CAMERA_HEIGHT = 1.7
LOS_MIN_BUFFER = 0.3
LOS_MAX_BUFFER = 2.0

# Facade length filtering parameters
MIN_SEGMENT_LENGTH = 3.0
MIN_FACADE_LENGTH_FOR_MATCHING = 3.0