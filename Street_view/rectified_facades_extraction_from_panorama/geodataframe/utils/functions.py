from geodataframe.utils.libraries import *

"""
Utility functions
"""

def auto_detect_zones(bucket_name: str = 'geolander.streetview', base_path="2025"):
    """Automatically detect available zones from the base path structure."""
    print(f"Auto-detecting zones from: {bucket_name}/{base_path}")
    
    try:
        s3_client_obj = S3Client()
        s3_client = s3_client_obj.get_s3_client()
        zones = []

        search_prefix = f"{base_path}/"
        paginator = s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket_name, Prefix=search_prefix, Delimiter='/'):
            # Ottieni tutti i prefissi comuni (equivalenti alle directory)
            for prefix in page.get('CommonPrefixes', []):
                # Estrai il nome della "directory" dal prefisso
                prefix_path = prefix.get('Prefix', '')
                zone_name = prefix_path.rstrip('/').split('/')[-1]
                
                # Verifica che la zona abbia la struttura corretta (contiene pano/ e shape/)
                pano_exists = False
                shape_exists = False
                
                # Controlla se esiste la sottocartella pano/
                pano_prefix = f"{prefix_path}pano/"
                pano_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=pano_prefix, MaxKeys=1)
                if 'Contents' in pano_response:
                    pano_exists = True
                
                # Controlla se esiste la sottocartella shape/
                shape_prefix = f"{prefix_path}shape/"
                shape_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=shape_prefix, MaxKeys=1)
                if 'Contents' in shape_response:
                    shape_exists = True
                
                # Aggiungi la zona solo se ha entrambe le cartelle richieste
                if pano_exists and shape_exists:
                    zones.append(zone_name)
        
        zones = sorted(list(set(zones)))
        print(f"Found {len(zones)} zones: {zones}")
        return zones
        
    except Exception as e:
        print(f"Error auto-detecting zones: {e}")
        return []

def auto_detect_run_values_in_zone(bucket_name: str = 'geolander.streetview', zone_path="2025/sudest2"):
    """Automatically detect available RUN values in a specific zone's pano folder."""
    print(f"Auto-detecting RUN values in zone: {zone_path}")
    
    try:
        s3_client_obj = S3Client()
        s3_client = s3_client_obj.get_s3_client()
        run_values = []
        
        # Cartella pano all'interno della zona
        pano_folder_prefix = f"{zone_path}/pano/"

        paginator = s3_client.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket_name, Prefix=pano_folder_prefix, Delimiter='/'):
            # Ottieni tutti i prefissi comuni (equivalenti alle directory)
            for prefix in page.get('CommonPrefixes', []):
                # Estrai il nome della "directory" dal prefisso
                prefix_path = prefix.get('Prefix', '')
                folder_name = prefix_path.rstrip('/').split('/')[-1]
                
                if folder_name.isdigit():
                    run_values.append(folder_name)
        
        run_values = sorted(list(set(run_values)))
        print(f"Found {len(run_values)} RUN values in zone {zone_path}: {run_values}")
        return run_values
        
    except Exception as e:
        print(f"Error auto-detecting RUN values in zone: {e}")
        return []

def setup_plotting_backend(interactive=False):
    """Setup matplotlib backend based on user preference."""
    if interactive:
        try:
            import matplotlib
            backends_to_try = ['Qt5Agg', 'TkAgg', 'Qt4Agg']
            
            for backend in backends_to_try:
                try:
                    matplotlib.use(backend)
                    print(f"Using interactive backend: {backend}")
                    break
                except ImportError:
                    continue
            else:
                print("No interactive backend available, falling back to file output")
                matplotlib.use('Agg')
        except Exception as e:
            print(f"Error setting up interactive plotting: {e}")
            plt.switch_backend('Agg')
    else:
        plt.switch_backend('Agg')
        print("Using non-interactive backend (file output only)")

def calculate_dynamic_bbox_from_cameras(camera_file_path, buffer_meters=200):
    """Calculate bounding box dynamically from camera positions with a buffer."""
    print(f"Loading camera data from: {camera_file_path}")
    
    try:
        s3_client = S3Client()
        gdf_camera = s3_client.read_shapefile('geolander.streetview', camera_file_path)
        print(f"Loaded {len(gdf_camera)} camera points")
        
        if gdf_camera.crs != "EPSG:4326":
            gdf_camera_wgs84 = gdf_camera.to_crs("EPSG:4326")
        else:
            gdf_camera_wgs84 = gdf_camera.copy()
        
        bounds = gdf_camera_wgs84.total_bounds
        buffer_degrees = buffer_meters / 111000  # Approximate conversion
        
        lat_min = bounds[1] - buffer_degrees
        lon_min = bounds[0] - buffer_degrees  
        lat_max = bounds[3] + buffer_degrees
        lon_max = bounds[2] + buffer_degrees
        
        print(f"Dynamic bounding box calculated:")
        print(f"  Latitude: {lat_min:.6f} to {lat_max:.6f}")
        print(f"  Longitude: {lon_min:.6f} to {lon_max:.6f}")
        print(f"  Buffer applied: {buffer_meters}m")
        
        return lat_min, lon_min, lat_max, lon_max
        
    except Exception as e:
        print(f"Error calculating dynamic bbox: {e}")
        return 44.805803, 10.328752, 44.807306, 10.332336

def filter_cameras_by_runs(gdf_camera, run_values=None):
    """Filter cameras by specific RUN values if provided."""
    if run_values is None:
        print("No RUN filter specified, using all camera points")
        return gdf_camera
    
    if 'RUN' not in gdf_camera.columns:
        print("Warning: 'RUN' column not found in camera data, using all points")
        return gdf_camera
    
    run_values_str = [str(rv) for rv in run_values]
    gdf_filtered = gdf_camera[gdf_camera['RUN'].astype(str).isin(run_values_str)]
    
    print(f"Filtered cameras from {len(gdf_camera)} to {len(gdf_filtered)} points")
    print(f"RUN values used: {run_values_str}")
    
    return gdf_filtered

def setup_output_directory(base_path="../data/Street_view/"):
    """Ensure output directory exists."""
    os.makedirs(base_path, exist_ok=True)
    return base_path

def filter_buildings_by_area(gdf_buildings, target_crs, percentile_threshold=10):
    """Filter buildings by area using percentile-based filtering."""
    if gdf_buildings.empty:
        print("No buildings to filter by area")
        return gdf_buildings
    
    gdf_projected = gdf_buildings.to_crs(target_crs)
    initial_count = len(gdf_projected)

    threshold = gdf_projected.geometry.area.quantile(percentile_threshold/100)
    print(f"\nApplying percentile-based area filtering:")
    print(f"  Removing smallest {percentile_threshold}% of buildings")
    print(f"  Area threshold: {threshold:.1f} m²")
    
    gdf_filtered = gdf_projected[gdf_projected.geometry.area >= threshold]
    removed_count = initial_count - len(gdf_filtered)
    
    print(f"  Buildings before filtering: {initial_count}")
    print(f"  Buildings after filtering: {len(gdf_filtered)}")
    print(f"  Buildings removed: {removed_count} ({removed_count/initial_count*100:.1f}%)")
    
    return gdf_filtered.to_crs("EPSG:4326")