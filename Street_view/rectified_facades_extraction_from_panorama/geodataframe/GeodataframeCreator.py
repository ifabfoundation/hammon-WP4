from geodataframe.utils.functions import *
from geodataframe.utils.camera import *
from geodataframe.utils.osm import *
from geodataframe.utils.core import *
from geodataframe.utils.libraries import *
from geodataframe.utils.facade import *
from geodataframe.settings import *

# Smart facade selection functions (defined here before use)
def has_clear_line_of_sight(facade_midpoint, camera_point, buildings_gdf, own_building_id, buffer_distance=0.5, check_radius=100):
    """
    Enhanced line-of-sight check with improved geometric reasoning.
    
    Key improvements:
    1. More sophisticated intersection analysis
    2. Considers building position relative to camera-facade line
    3. Excludes buildings that are behind the camera or facade
    4. Uses adaptive intersection thresholds
    5. Better handling of edge cases
    
    Parameters:
    -----------
    facade_midpoint : Point
        The midpoint of the facade
    camera_point : Point  
        The camera position
    buildings_gdf : GeoDataFrame
        All buildings in the area
    own_building_id : str/int
        ID of the building that owns this facade (always excluded)
    buffer_distance : float
        Buffer around the sight line for intersection checking
    check_radius : float
        Maximum distance from facade midpoint to check for blocking buildings (meters)
        
    Returns:
    --------
    bool : True if line of sight is clear, False if blocked
    """
    # Create sight line from camera to facade
    sight_line = LineString([(camera_point.x, camera_point.y), (facade_midpoint.x, facade_midpoint.y)])
    sight_line_length = sight_line.length
    
    # Adaptive buffer - smaller for very close facades, larger for distant ones
    adaptive_buffer = min(max(buffer_distance, sight_line_length * 0.01), 2.0)  # 1% of distance, capped at 2m
    sight_line_buffered = sight_line.buffer(adaptive_buffer)
    
    # Create search area - use smaller radius for closer facades to improve performance
    effective_radius = min(check_radius, sight_line_length * 1.5)  # Don't search beyond 1.5x the sight line length
    search_area = facade_midpoint.buffer(effective_radius)
    
    # Pre-filter buildings to only those within the search radius
    try:
        if hasattr(buildings_gdf, 'sindex'):
            possible_matches_index = list(buildings_gdf.sindex.intersection(search_area.bounds))
            nearby_buildings = buildings_gdf.iloc[possible_matches_index]
        else:
            nearby_buildings = buildings_gdf[buildings_gdf.geometry.intersects(search_area)]
    except Exception:
        building_centroids = buildings_gdf.geometry.centroid
        distances = building_centroids.distance(facade_midpoint)
        nearby_buildings = buildings_gdf[distances <= effective_radius]
    
    # Direction vector from camera to facade
    camera_to_facade = np.array([facade_midpoint.x - camera_point.x, facade_midpoint.y - camera_point.y])
    camera_to_facade_norm = camera_to_facade / (np.linalg.norm(camera_to_facade) + 1e-10)
    
    # Check each nearby building for blocking
    for _, building in nearby_buildings.iterrows():
        if building['osm_id'] == own_building_id:
            continue
            
        try:
            building_geometry = building.geometry
            
            # Quick check: if building doesn't intersect buffered sight line, skip detailed analysis
            if not sight_line_buffered.intersects(building_geometry):
                continue
            
            # Get building centroid for position analysis
            building_centroid = building_geometry.centroid
            camera_to_building = np.array([building_centroid.x - camera_point.x, building_centroid.y - camera_point.y])
            
            # Skip buildings that are behind the camera (dot product < 0)
            if np.dot(camera_to_building, camera_to_facade_norm) < 0:
                continue
            
            # Skip buildings that are beyond the facade (project building position onto sight line)
            # If the projection is beyond the facade point, the building is behind the facade
            projection_length = np.dot(camera_to_building, camera_to_facade_norm)
            if projection_length > sight_line_length * 1.1:  # Allow 10% tolerance
                continue
            
            # Detailed intersection analysis
            intersection = sight_line_buffered.intersection(building_geometry)
            
            if intersection.is_empty:
                continue
            
            # Calculate intersection significance
            if hasattr(intersection, 'area'):
                # For polygon intersections
                intersection_area = intersection.area
                
                # Adaptive threshold based on building size and distance
                building_area = building_geometry.area
                distance_to_camera = np.linalg.norm(camera_to_building)
                
                # Buildings closer to camera need smaller intersection to block view
                # Larger buildings need larger intersection to be considered blocking
                distance_factor = max(0.1, distance_to_camera / 50.0)  # Normalize by 50m
                area_factor = max(0.1, building_area / 100.0)  # Normalize by 100m²
                
                # Dynamic threshold: smaller for close/large buildings, larger for distant/small ones
                area_threshold = 0.05 * distance_factor * area_factor
                area_threshold = max(0.01, min(area_threshold, 1.0))  # Clamp between 0.01 and 1.0
                
                if intersection_area > area_threshold:
                    return False
                    
            elif hasattr(intersection, 'length'):
                # For line intersections
                intersection_length = intersection.length
                
                # Calculate what percentage of the sight line is blocked
                if sight_line_length > 0:
                    blocked_percentage = intersection_length / sight_line_length
                    
                    # If more than 5% of the sight line is blocked, consider it obstructed
                    if blocked_percentage > 0.05:
                        return False
                        
            else:
                # For point intersections or other geometries
                # Check if it's a substantial intersection (not just touching)
                if not intersection.is_empty:
                    # For point intersections, check if multiple points or significant touching
                    try:
                        # Convert to string and check for substantial intersection
                        intersection_str = str(intersection)
                        if 'POLYGON' in intersection_str or 'LINESTRING' in intersection_str:
                            return False
                    except:
                        # If we can't determine the nature, be conservative and allow
                        continue
                        
        except Exception as e:
            # If any error occurs with this building, skip it but continue checking others
            continue
    
    return True

def enhanced_line_of_sight_with_height(facade_midpoint, camera_point, buildings_gdf, own_building_id, 
                                    facade_height=3.0, camera_height=1.7, check_radius=100):
    """
    Advanced line-of-sight check that considers building heights.
    This is an optional enhanced version for cases where building height data is available.
    
    Parameters:
    -----------
    facade_midpoint : Point
        The midpoint of the facade
    camera_point : Point  
        The camera position
    buildings_gdf : GeoDataFrame
        All buildings in the area (should have 'height' column if available)
    own_building_id : str/int
        ID of the building that owns this facade
    facade_height : float
        Height of the facade being viewed (meters)
    camera_height : float
        Height of the camera above ground (meters)
    check_radius : float
        Maximum distance to check for blocking buildings (meters)
        
    Returns:
    --------
    bool : True if line of sight is clear, False if blocked
    """
    # First run the standard 2D check
    if not has_clear_line_of_sight(facade_midpoint, camera_point, buildings_gdf, own_building_id, check_radius=check_radius):
        return False
    
    # If building height data is not available, fall back to 2D check
    if 'height' not in buildings_gdf.columns:
        return True
    
    # 3D line of sight check
    sight_line_2d = LineString([(camera_point.x, camera_point.y), (facade_midpoint.x, facade_midpoint.y)])
    sight_distance = sight_line_2d.length
    
    # Calculate the viewing angle (elevation angle)
    height_difference = facade_height - camera_height
    viewing_angle = np.arctan2(height_difference, sight_distance)
    
    # Get buildings that intersect the 2D sight line
    sight_line_buffered = sight_line_2d.buffer(1.0)  # 1m buffer for 3D check
    
    try:
        if hasattr(buildings_gdf, 'sindex'):
            search_area = facade_midpoint.buffer(check_radius)
            possible_matches_index = list(buildings_gdf.sindex.intersection(search_area.bounds))
            nearby_buildings = buildings_gdf.iloc[possible_matches_index]
        else:
            nearby_buildings = buildings_gdf[buildings_gdf.geometry.distance(facade_midpoint) <= check_radius]
    except Exception:
        return True  # If spatial operations fail, be conservative
    
    for _, building in nearby_buildings.iterrows():
        if building['osm_id'] == own_building_id:
            continue
            
        try:
            if not sight_line_buffered.intersects(building.geometry):
                continue
                
            # Get building height
            building_height = building.get('height', 0)
            if pd.isna(building_height) or building_height <= 0:
                building_height = 10  # Default assumption for buildings without height data
                
            # Calculate distance from camera to this building
            building_centroid = building.geometry.centroid
            distance_to_building = Point(camera_point.x, camera_point.y).distance(building_centroid)
            
            # Calculate the angle to the top of the blocking building
            blocking_angle = np.arctan2(building_height - camera_height, distance_to_building)
            
            # If the blocking building's top is higher than our line of sight to the facade, it blocks the view
            if blocking_angle > viewing_angle:
                return False
                
        except Exception:
            continue
    
    return True

def smart_line_of_sight_check(facade_midpoint, camera_point, buildings_gdf, own_building_id, 
                            facade_height=None, check_radius=100):
    """
    Smart wrapper for line-of-sight checking that automatically selects the best method
    based on available data and configuration options.
    
    Parameters:
    -----------
    facade_midpoint : Point
        The midpoint of the facade
    camera_point : Point  
        The camera position
    buildings_gdf : GeoDataFrame
        All buildings in the area
    own_building_id : str/int
        ID of the building that owns this facade
    facade_height : float, optional
        Height of the facade being viewed (meters). If None, uses DEFAULT_FACADE_HEIGHT
    check_radius : float
        Maximum distance to check for blocking buildings (meters)
        
    Returns:
    --------
    bool : True if line of sight is clear, False if blocked
    """
    
    # Calculate adaptive buffer based on distance
    sight_distance = Point(camera_point.x, camera_point.y).distance(facade_midpoint)
    buffer_distance = min(max(LOS_MIN_BUFFER, sight_distance * 0.01), LOS_MAX_BUFFER)

        
    return has_clear_line_of_sight(
        facade_midpoint, camera_point, buildings_gdf, own_building_id,
        buffer_distance=buffer_distance,
        check_radius=check_radius
    )

def debug_line_of_sight_comparison(facade_midpoint, camera_point, buildings_gdf, own_building_id, 
                                check_radius=100, verbose=True):
    """
    Debug function to compare old vs new line-of-sight methods and analyze differences.
    
    Returns:
    --------
    dict : Comparison results with detailed analysis
    """
    
    # Test with original simple method
    sight_line = LineString([(camera_point.x, camera_point.y), (facade_midpoint.x, facade_midpoint.y)])
    sight_line_buffered = sight_line.buffer(0.5)
    
    old_method_blocked = False
    old_method_blocking_buildings = []
    
    for _, building in buildings_gdf.iterrows():
        if building['osm_id'] == own_building_id:
            continue
        try:
            if sight_line_buffered.intersects(building.geometry):
                intersection = sight_line_buffered.intersection(building.geometry)
                if hasattr(intersection, 'area') and intersection.area > 0.1:
                    old_method_blocked = True
                    old_method_blocking_buildings.append(building['osm_id'])
                elif not hasattr(intersection, 'area') and not intersection.is_empty:
                    old_method_blocked = True
                    old_method_blocking_buildings.append(building['osm_id'])
        except Exception:
            continue
    
    # Test with enhanced method
    new_method_blocked = not has_clear_line_of_sight(
        facade_midpoint, camera_point, buildings_gdf, own_building_id, 
        buffer_distance=0.5, check_radius=check_radius
    )
    
    # Test with smart wrapper (using current configuration)
    smart_method_blocked = not smart_line_of_sight_check(
        facade_midpoint, camera_point, buildings_gdf, own_building_id, 
        check_radius=check_radius
    )
    
    results = {
        'sight_line_length': sight_line.length,
        'old_method_blocked': old_method_blocked,
        'new_method_blocked': new_method_blocked,
        'smart_method_blocked': smart_method_blocked,
        'old_blocking_buildings': old_method_blocking_buildings,
        'methods_agree': old_method_blocked == new_method_blocked == smart_method_blocked,
        'old_vs_new_agree': old_method_blocked == new_method_blocked,
        'old_vs_smart_agree': old_method_blocked == smart_method_blocked,
        'new_vs_smart_agree': new_method_blocked == smart_method_blocked
    }
    
    if verbose:
        print(f"\nLine-of-sight comparison for facade-camera distance: {sight_line.length:.2f}m")
        print(f"  Old method (simple): {'BLOCKED' if old_method_blocked else 'CLEAR'}")
        print(f"  New method (enhanced): {'BLOCKED' if new_method_blocked else 'CLEAR'}")
        print(f"  Smart method (configured): {'BLOCKED' if smart_method_blocked else 'CLEAR'}")
        
        if old_method_blocking_buildings:
            print(f"  Old method blocking buildings: {old_method_blocking_buildings}")
        
        if not results['methods_agree']:
            print(f"  ⚠ Methods disagree!")
        else:
            print(f"  ✓ All methods agree")
    
    return results

def filter_facades_by_minimum_length(facades_gdf, min_length=1):
    print(f"Filtering facades by minimum length ({min_length}m)...")
    if facades_gdf.empty:
        print("No facades to filter by length.")
        return facades_gdf 
    initial_count = len(facades_gdf)
    filtered_gdf = facades_gdf[facades_gdf['length'] >= min_length].copy()
    print(f"Facades after length filter: {len(filtered_gdf)} (removed {initial_count - len(filtered_gdf)} small facades)")
    return filtered_gdf

def is_camera_in_facade_field_of_view(facade_midpoint, facade_segment, camera_point, facade_orientation, fov_degrees=45):
    coords = list(facade_segment.coords)
    p1 = np.array(coords[0]); p2 = np.array(coords[1])
    facade_direction = p2 - p1
    if np.linalg.norm(facade_direction) < 1e-9: return False # Avoid division by zero for zero-length segment
    facade_direction_norm = facade_direction / np.linalg.norm(facade_direction)
    
    normal_candidate1 = np.array([-facade_direction_norm[1], facade_direction_norm[0]])
    normal_candidate2 = np.array([facade_direction_norm[1], -facade_direction_norm[0]])
    
    expected_normals = {'N': np.array([0, 1]), 'S': np.array([0, -1]), 'E': np.array([1, 0]), 'W': np.array([-1, 0])}
    expected_normal = expected_normals.get(facade_orientation, np.array([0,0])) # Default to avoid error if orientation is unexpected

    dot1 = np.dot(normal_candidate1, expected_normal)
    dot2 = np.dot(normal_candidate2, expected_normal)
    facade_normal = normal_candidate1 if dot1 >= dot2 else normal_candidate2 # Use >= for consistency
    
    to_camera = np.array([camera_point.x - facade_midpoint.x, camera_point.y - facade_midpoint.y])
    if np.linalg.norm(to_camera) < 1e-9: return True # Camera is at midpoint, technically in FOV
    to_camera_norm = to_camera / np.linalg.norm(to_camera)
    
    dot_product = np.dot(facade_normal, to_camera_norm)
    dot_product = np.clip(dot_product, -1, 1)
    # angle_rad = np.arccos(abs(dot_product)) # This gives angle to the line containing normal
    # We need angle to the normal vector itself. Facade normal should point towards camera.
    # If dot_product is positive, camera is in front. If negative, behind.
    # We want the camera to be in front of the facade.
    if dot_product < 0: # Camera is behind the facade normal
        return False 

    angle_rad_to_normal_direction = np.arccos(dot_product) # Angle between facade_normal and to_camera_norm
    angle_deg = np.degrees(angle_rad_to_normal_direction)
    
    return angle_deg <= (fov_degrees / 2) # Check if within half of FOV from normal

def validate_camera_facade_direction(facade_midpoint, facade_segment, camera_point, facade_orientation):
    """
    Simple validation that camera is within ±45° of the facade normal direction.
    
    Parameters:
    -----------
    facade_midpoint : Point
        Midpoint of the facade
    facade_segment : LineString
        The facade geometry
    camera_point : Point
        Camera position
    facade_orientation : str
        Facade orientation (N, S, E, W) - not strictly used
    debug : bool
        Print debug information
        
    Returns:
    --------
    bool : True if camera is within acceptable angle of facade normal
    """
    
    # Vector from facade to camera
    to_camera = np.array([camera_point.x - facade_midpoint.x, camera_point.y - facade_midpoint.y])
    if np.linalg.norm(to_camera) < 1e-9: 
        return True # Camera at facade midpoint
    to_camera_norm = to_camera / np.linalg.norm(to_camera)
    
    # Get facade direction vector
    coords = list(facade_segment.coords)
    p1 = np.array(coords[0])
    p2 = np.array(coords[1])
    facade_direction = p2 - p1
    if np.linalg.norm(facade_direction) < 1e-9:
        return False # Zero-length facade
    facade_direction_norm = facade_direction / np.linalg.norm(facade_direction)
    
    # Calculate both possible normals (perpendicular to facade)
    normal_right = np.array([-facade_direction_norm[1], facade_direction_norm[0]])  # 90° clockwise
    normal_left = np.array([facade_direction_norm[1], -facade_direction_norm[0]])   # 90° counter-clockwise
    
    # Choose the normal that points more towards the camera
    dot_right = np.dot(normal_right, to_camera_norm)
    dot_left = np.dot(normal_left, to_camera_norm)
    
    if dot_right > dot_left:
        facade_normal = normal_right
        alignment = dot_right
    else:
        facade_normal = normal_left
        alignment = dot_left
    
    # Check if camera is within ±45° of the facade normal
    # cos(45°) ≈ 0.707, so we need alignment > 0.707
    min_alignment = 0.707  # 45 degrees
    
    if alignment < min_alignment:
        return False
    
    return True

def is_camera_in_correct_direction(facade_midpoint, camera_point, facade_orientation):
    """
    Simplified direction checking - just ensure camera is generally in front of facade.
    """
    # This is now just a basic check - the main validation is in validate_camera_facade_direction
    return True  # Let the main validation function handle this

def is_camera_viewing_facade_correctly(facade_midpoint, facade_segment, camera_point, facade_orientation, buildings_gdf=None, own_building_id=None, los_radius=LINE_OF_SIGHT_RADIUS):
    """
    Simplified facade-camera validation using ±45° normal check.
    """
    
    # Step 1: Simple direction validation (±45° from facade normal)
    if not validate_camera_facade_direction(facade_midpoint, facade_segment, camera_point, facade_orientation):
        return False
    
    # Step 2: Enhanced line of sight check
    if buildings_gdf is not None and own_building_id is not None:
        if not smart_line_of_sight_check(facade_midpoint, camera_point, buildings_gdf, own_building_id, check_radius=los_radius):
            return False
    
    return True

def calculate_viewing_quality(facade_segment, camera_point):
    facade_midpoint = facade_segment.centroid
    to_facade = np.array([facade_midpoint.x - camera_point.x, facade_midpoint.y - camera_point.y])
    if np.linalg.norm(to_facade) < 1e-9: return 1.0 # Max quality if at same point
    to_facade_norm = to_facade / np.linalg.norm(to_facade)
    
    coords = list(facade_segment.coords)
    facade_direction = np.array([coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]])
    if np.linalg.norm(facade_direction) < 1e-9: return 0.0 # No defined direction
    facade_direction_norm = facade_direction / np.linalg.norm(facade_direction)
    
    # Normal pointing towards camera
    normal_cand1 = np.array([-facade_direction_norm[1], facade_direction_norm[0]])
    normal_cand2 = np.array([facade_direction_norm[1], -facade_direction_norm[0]])
    if np.dot(normal_cand1, to_facade_norm) > np.dot(normal_cand2, to_facade_norm):
        facade_normal_towards_camera = normal_cand1
    else:
        facade_normal_towards_camera = normal_cand2

    # Quality is how aligned 'to_facade_norm' is with 'facade_normal_towards_camera'
    # Perfect alignment (perpendicular view) means dot product is 1
    quality = np.dot(to_facade_norm, facade_normal_towards_camera)
    quality = np.clip(quality, 0, 1) # Ensure quality is between 0 and 1
    return quality

# This is the ACTIVE enhanced_facade_camera_matching
def enhanced_facade_camera_matching(facades_gdf, cameras_gdf, buildings_gdf, max_distance=50, los_radius=LINE_OF_SIGHT_RADIUS):
    print(f"Enhanced matching of {len(facades_gdf)} facades with {len(cameras_gdf)} cameras...")
    print(f"Using ±45° facade normal validation and line-of-sight checking (radius: {los_radius}m)")
    
    matched_facades_list = []
    unmatched_facades_list = []
    camera_usage = {}
    
    # Statistics tracking
    direction_failed_count = 0
    line_of_sight_blocked_count = 0
    total_los_checks = 0
    total_direction_checks = 0
    
    if facades_gdf.empty or cameras_gdf.empty:
        print("No facades or cameras to match.")
        return matched_facades_list, unmatched_facades_list, camera_usage

    facade_priorities = []
    for idx, facade in facades_gdf.iterrows():
        min_dist = cameras_gdf.geometry.distance(facade['geometry_midpoint']).min()
        facade_priorities.append((min_dist, idx))
    facade_priorities.sort()
    
    print(f"Processing {len(facade_priorities)} facades in order of distance to nearest camera...")
    
    for _, facade_idx in tqdm(facade_priorities, desc="Matching facades to cameras"):
        facade = facades_gdf.loc[facade_idx]
        facade_midpoint = facade['geometry_midpoint']
        facade_segment = facade['geometry_segment']
        facade_orientation = facade['orientation']
        
        distances = cameras_gdf.geometry.distance(facade_midpoint)
        current_max_distance = max_distance * 1.5 if facade_orientation in ['E', 'W'] else max_distance
        sorted_camera_indices = distances.sort_values().index
        
        camera_found_for_facade = False
        for cam_idx in sorted_camera_indices:
            camera_point = cameras_gdf.loc[cam_idx].geometry
            distance_to_camera = distances.loc[cam_idx]
            
            if distance_to_camera > current_max_distance: 
                break
            
            max_facades_per_camera = 5 
            if cam_idx in camera_usage and camera_usage[cam_idx] >= max_facades_per_camera: 
                continue
            
            # Simple direction validation (±45° from facade normal)
            total_direction_checks += 1
            if not validate_camera_facade_direction(facade_midpoint, facade_segment, camera_point, facade_orientation):
                direction_failed_count += 1
                continue
            
            # CRITICAL CHECK: Ensure sight line doesn't pass through facade's own building
            own_building_matches = buildings_gdf[buildings_gdf['osm_id'] == facade['building_id']]
            own_building_polygon = own_building_matches.iloc[0].geometry
            sight_line = LineString([(camera_point.x, camera_point.y), (facade_midpoint.x, facade_midpoint.y)])
            
            # Buffer the building inward slightly to avoid edge effects
            try:
                interior_polygon = own_building_polygon.buffer(-0.1)  # 10cm inward
                if interior_polygon.is_empty:
                    interior_polygon = own_building_polygon
                
                # Check if sight line passes through building interior
                if sight_line.intersects(interior_polygon):
                    intersection = sight_line.intersection(interior_polygon)
                    
                    # Check if it's a substantial intersection (not just touching edges)
                    is_substantial = False
                    if hasattr(intersection, 'length') and intersection.length > 0.5:  # >50cm
                        is_substantial = True
                    elif hasattr(intersection, 'area') and intersection.area > 0.1:  # >0.1m²
                        is_substantial = True
                    elif not intersection.is_empty:
                        try:
                            intersection_str = str(intersection)
                            if 'LINESTRING' in intersection_str or 'POLYGON' in intersection_str:
                                is_substantial = True
                        except:
                            pass
                    
                    if is_substantial:
                        direction_failed_count += 1  # Count as direction failure
                        continue
                        
            except Exception as e:
                # If geometric operations fail, be conservative and allow the match
                pass
            
            # Line of sight check (for OTHER buildings blocking the view)
            if buildings_gdf is not None and facade['building_id'] is not None:
                total_los_checks += 1
                if not smart_line_of_sight_check(facade_midpoint, camera_point, buildings_gdf, facade['building_id'], check_radius=los_radius):
                    line_of_sight_blocked_count += 1
                    continue
            
            # All checks passed - create match
            facade_dict = facade.to_dict()
            facade_dict['matched_camera_idx'] = cam_idx
            facade_dict['camera_distance'] = distance_to_camera
            facade_dict['camera_geometry'] = camera_point
            facade_dict['viewing_quality'] = calculate_viewing_quality(facade_segment, camera_point)
            matched_facades_list.append(facade_dict)
            camera_usage[cam_idx] = camera_usage.get(cam_idx, 0) + 1
            camera_found_for_facade = True
            break
        
        if not camera_found_for_facade:
            unmatched_dict = facade.to_dict()
            unmatched_dict['reason'] = f'No suitable camera found (max_dist: {current_max_distance:.1f}m)'
            unmatched_facades_list.append(unmatched_dict)
            
    print(f"Simplified matching results:")
    print(f"  Matched: {len(matched_facades_list)} facades")
    print(f"  Unmatched: {len(unmatched_facades_list)} facades")
    print(f"  Direction validation checks: {total_direction_checks}")
    print(f"  Direction validation failures: {direction_failed_count} ({direction_failed_count/max(1,total_direction_checks)*100:.1f}%)")
    print(f"    (includes facades with sight lines passing through own building)")
    print(f"  Line-of-sight checks performed: {total_los_checks}")
    print(f"  Line-of-sight blocked: {line_of_sight_blocked_count} ({line_of_sight_blocked_count/max(1,total_los_checks)*100:.1f}%)")
    print(f"  Cameras used: {len(camera_usage)} out of {len(cameras_gdf)}")
    
    if matched_facades_list:
        matched_df = pd.DataFrame(matched_facades_list)
        print("  Matched by orientation:")
        print(matched_df['orientation'].value_counts().to_string())
    
    return matched_facades_list, unmatched_facades_list, camera_usage

def apply_enhanced_matching(facades_to_match_gdf, cameras_subset_gdf, all_buildings_gdf, min_facade_len=1, los_radius=LINE_OF_SIGHT_RADIUS):
    facades_len_filtered_gdf = filter_facades_by_minimum_length(
        facades_to_match_gdf, 
        min_length=min_facade_len
    )
    
    matched_facades_list, unmatched_facades_list, cam_usage_dict = enhanced_facade_camera_matching(
        facades_len_filtered_gdf, 
        cameras_subset_gdf,
        all_buildings_gdf,
        max_distance=50,
        los_radius=los_radius
    )
    
    # Define columns based on facades_len_filtered_gdf for consistency, add new ones
    base_cols = list(facades_len_filtered_gdf.columns) if not facades_len_filtered_gdf.empty else []
    # Ensure geometry column name is derived correctly
    geom_col_name = 'geometry_segment' if 'geometry_segment' in base_cols else (facades_len_filtered_gdf.geometry.name if hasattr(facades_len_filtered_gdf,'geometry') and facades_len_filtered_gdf.geometry.name else 'geometry')


    final_matched_cols = base_cols + ['matched_camera_idx', 'camera_distance', 'camera_geometry', 'viewing_quality']
    final_unmatched_cols = base_cols + ['reason']

    # Create GeoDataFrames carefully
    if matched_facades_list:
        matched_gdf = gpd.GeoDataFrame(matched_facades_list, columns=final_matched_cols, geometry=geom_col_name, crs=facades_to_match_gdf.crs)
    else:
        # Create empty GDF with correct schema
        temp_df_matched = pd.DataFrame(columns=[col for col in final_matched_cols if col != geom_col_name])
        matched_gdf = gpd.GeoDataFrame(temp_df_matched, geometry=gpd.GeoSeries([], name=geom_col_name, crs=facades_to_match_gdf.crs), crs=facades_to_match_gdf.crs)


    if unmatched_facades_list:
        unmatched_gdf = gpd.GeoDataFrame(unmatched_facades_list, columns=final_unmatched_cols, geometry=geom_col_name, crs=facades_to_match_gdf.crs)
    else:
        temp_df_unmatched = pd.DataFrame(columns=[col for col in final_unmatched_cols if col != geom_col_name])
        unmatched_gdf = gpd.GeoDataFrame(temp_df_unmatched, geometry=gpd.GeoSeries([], name=geom_col_name, crs=facades_to_match_gdf.crs), crs=facades_to_match_gdf.crs)

    return matched_gdf, unmatched_gdf, cam_usage_dict

def create_geodataframe():
    # Setup plotting backend
    setup_plotting_backend(interactive=INTERACTIVE_PLOTS)

    # Setup output directory
    output_dir = setup_output_directory(OUTPUT_DIR)

    # Determine RUN values to use
    RUN_VALUES = auto_detect_run_values(PANO_FOLDER_PATH)

    # Calculate dynamic bounding box from camera positions
    lat_min, lon_min, lat_max, lon_max = calculate_dynamic_bbox_from_cameras(
        CAMERA_FILE_PATH, 
        buffer_meters=BUFFER_METERS
    )

    # Query OSM and process building data
    print("Querying OpenStreetMap data...")
    api = overpy.Overpass()
    query = f"""
    [out:json][timeout:180];
    (
    way["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    relation["building"]({lat_min},{lon_min},{lat_max},{lon_max});
    );
    out body;
    >;
    out skel qt;
    """
    result = api.query(query)
    buildings = []
    for element in result.ways + result.relations:
        if 'building' in element.tags:
            building_data = process_building(element)
            if building_data:
                buildings.append(building_data)
    print(f"Found {len(buildings)} buildings")

    # Create GeoDataFrame and process buildings
    gdf = gpd.GeoDataFrame(buildings, crs="EPSG:4326")
    if not gdf.empty:
        gdf = gdf[gdf.geometry.is_valid]

    gdf = filter_buildings_by_area(gdf, TARGET_CRS, percentile_threshold=AREA_PERCENTILE_THRESHOLD)

    gdf_projected = gdf.to_crs(TARGET_CRS)
    print(f"Buildings ready for processing: {len(gdf_projected)}")

    # Group buildings and find exterior boundaries
    print("\nGrouping adjacent buildings and finding exterior boundaries...")
    buffer_distance = 0.1

    if not gdf_projected.empty:
        gdf_projected['temp_id'] = gdf_projected.index
        gdf_buffered = gdf_projected.copy()
        gdf_buffered['geometry'] = gdf_projected.geometry.buffer(buffer_distance)
        joined_gdf = gpd.sjoin(gdf_buffered, gdf_buffered, how='inner', predicate='intersects')
        G = nx.Graph()
        for _, row in joined_gdf.iterrows():
            G.add_edge(row['temp_id_left'], row['temp_id_right'])
        connected_components = list(nx.connected_components(G))
        gdf_projected['group_id'] = None
        for group_id_num, component in enumerate(connected_components):
            if len(component) == 1:
                gdf_projected.loc[list(component), 'group_id'] = f'standalone_{group_id_num}'
            else:
                gdf_projected.loc[list(component), 'group_id'] = f'block_{group_id_num}'
    else:
        print("No projected buildings to group.")
        gdf_projected['group_id'] = None

    # Process each group to find exterior boundaries and facade segments
    print("\nExtracting facades from exterior boundaries...")
    facade_segments = []
    exterior_boundaries = []

    if not gdf_projected.empty and 'group_id' in gdf_projected.columns and gdf_projected['group_id'].notna().any():
        for group_id, group_gdf in tqdm(gdf_projected.groupby('group_id'), desc="Processing building groups"):
            merged_polygon = unary_union(group_gdf['geometry'])
            polygons = list(merged_polygon.geoms) if merged_polygon.geom_type == 'MultiPolygon' else [merged_polygon]
            
            for polygon in polygons:
                if not polygon.is_valid: 
                    polygon = make_valid(polygon)
                if polygon.is_empty or polygon.geom_type != 'Polygon': 
                    continue

                exterior_boundary = polygon.exterior
                exterior_boundaries.append({'group_id': group_id, 'geometry': exterior_boundary})
                exterior_coords = list(exterior_boundary.coords)
                
                for i in range(len(exterior_coords) - 1):
                    p1, p2 = exterior_coords[i], exterior_coords[i + 1]
                    segment = LineString([p1, p2])
                    if segment.length >= MIN_SEGMENT_LENGTH:
                        segment_midpoint = segment.centroid
                        min_dist_to_building_boundary = float('inf')
                        closest_building_id_for_segment = None
                        
                        for _, building in group_gdf.iterrows():
                            if segment.distance(building.geometry.exterior) < 0.01:
                                dist_midpoint_to_poly = building.geometry.distance(segment_midpoint)
                                if dist_midpoint_to_poly < min_dist_to_building_boundary:
                                    min_dist_to_building_boundary = dist_midpoint_to_poly
                                    closest_building_id_for_segment = building['osm_id']

                        if closest_building_id_for_segment is not None:
                            facade_segments.append({
                                'group_id': group_id,
                                'building_id': closest_building_id_for_segment,
                                'geometry_segment': segment,
                                'geometry_midpoint': segment_midpoint,
                                'length': segment.length,
                                'is_exterior': True
                            })
    else:
        print("No building groups to process for facade extraction.")

    facade_segments_gdf = gpd.GeoDataFrame(facade_segments, geometry='geometry_segment', crs=TARGET_CRS)
    exterior_boundaries_gdf = gpd.GeoDataFrame(exterior_boundaries, geometry='geometry', crs=TARGET_CRS)
    print(f"Found {len(facade_segments_gdf)} exterior facade segments")

    # Add orientation to facades
    facade_segments_gdf = add_orientation_to_facades(facade_segments_gdf, gdf_projected)

    # Load and filter camera points using dynamic functions
    print("\nLoading and filtering camera points...")

    def calculate_orientation(dx, dy):
        """
        Calculate orientation from dx, dy differences
        Returns cardinal direction (N, S, E, W)
        """
        # Calculate angle in radians
        orientation_rad = math.atan2(dy, dx)
        # Convert to degrees
        orientation_degrees = math.degrees(orientation_rad)
        # Convert to azimuth (0-360, where 0 is East, 90 is North)
        # We need to adjust this to geographic convention (0 is North, 90 is East)
        azimuth = (90 - orientation_degrees) % 360
        
        # Determine cardinal direction
        if azimuth >= 315 or azimuth < 45:
            return 'N'
        elif 45 <= azimuth < 135:
            return 'E'
        elif 135 <= azimuth < 225:
            return 'S'
        elif 225 <= azimuth < 315:
            return 'W'
        else:
            return 'N'  # fallback
    
    gdf_camera_all = gpd.read_file(CAMERA_FILE_PATH)
    
    # Main processing loop
    for index, row in gdf_camera_all.iterrows():
        foto = row['FOTO'].split('.')[0]
        foto_number = int(foto[-6:])
        orientation = None
        
        try:
            # First, try to use the next photo for orientation
            if index + 1 < len(gdf_camera_all):
                next_row = gdf_camera_all.iloc[index + 1]
                foto_number_next = int(next_row['FOTO'].split('.')[0][-6:])
                
                # Check if next photo is consecutive
                if foto_number_next - foto_number == 1:
                    dx = next_row['X'] - row['X']
                    dy = next_row['Y'] - row['Y']
                    orientation = calculate_orientation(dx, dy)
            
            # If we couldn't use next photo, try previous photo
            if orientation is None and index > 0:
                previous_row = gdf_camera_all.iloc[index - 1]
                foto_number_previous = int(previous_row['FOTO'].split('.')[0][-6:])
                
                # Check if current photo is consecutive after previous
                if foto_number - foto_number_previous == 1:
                    dx = row['X'] - previous_row['X']
                    dy = row['Y'] - previous_row['Y']
                    orientation = calculate_orientation(dx, dy)
            
            # Assign orientation if we found one
            if orientation is not None:
                gdf_camera_all.loc[index, 'orientation_run'] = orientation
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # Optional: Fill any remaining NaN values with a default or interpolate
    print("Orientation calculation completed!")
    print(f"Rows with orientation: {gdf_camera_all['orientation_run'].notna().sum()}")
    print(f"Rows without orientation: {gdf_camera_all['orientation_run'].isna().sum()}")
    print(gdf_camera_all.columns)

    gdf_camera_subset = filter_cameras_by_runs(gdf_camera_all, RUN_VALUES)
    gdf_camera_subset = gdf_camera_subset.to_crs(TARGET_CRS)
    print(f"Final camera points for matching: {len(gdf_camera_subset)}")
    print(gdf_camera_subset.columns)

    # Filter for street-facing facades
    facade_segments_filtered = filter_street_facing_facades(
        facade_segments_gdf, 
        gdf_camera_subset,
        buildings_gdf=gdf_projected,
        max_distance=30
    )

    # Filter for only N, S, E, W facing facades
    if not facade_segments_filtered.empty:
        facade_segments_filtered = facade_segments_filtered[
            facade_segments_filtered['orientation'].isin(['N', 'S', 'E', 'W'])
        ]
    print(f"N/S/E/W facing facades: {len(facade_segments_filtered)}")

    # Merge adjacent facade segments
    facade_segments_merged = merge_adjacent_facade_segments(
        facade_segments_filtered,
        angle_tolerance=15,
        distance_tolerance=2
    )

    # Smart facade selection based on urban vs suburban context
    print("\nAnalyzing building density distribution...")
    suggested_threshold = analyze_building_density_distribution(facade_segments_merged)

    if suggested_threshold and suggested_threshold != BUILDING_DENSITY_THRESHOLD:
        print(f"Note: You might want to consider using threshold={suggested_threshold} for your area")

    print(f"\nApplying enhanced smart facade selection...")

    # Apply force single facade option if enabled
    effective_threshold = BUILDING_DENSITY_THRESHOLD

    best_facades_gdf = smart_facade_selection_by_context(
        facade_segments_merged, 
        gdf_camera_subset, 
        building_density_threshold=effective_threshold
    )
            
    print(f"Final facades selected for matching: {len(best_facades_gdf)}")

    # Enhanced matching call
    print(f"\nApplying enhanced facade-camera matching with all improvements...")
    print(f"Line-of-sight optimization: checking buildings within {LINE_OF_SIGHT_RADIUS}m radius only")
    matched_facades_gdf, unmatched_facades_gdf, camera_usage = apply_enhanced_matching(
        best_facades_gdf, 
        gdf_camera_subset,
        gdf_projected, # Pass all buildings for LoS
        min_facade_len=MIN_FACADE_LENGTH_FOR_MATCHING,
        los_radius=LINE_OF_SIGHT_RADIUS
    )
    print(f"\nMatching results:")
    print(f"Facades with suitable camera: {len(matched_facades_gdf)}")
    print(f"Facades without suitable camera: {len(unmatched_facades_gdf)}")


    # Create final combined dataset only for matched facades
    print("\nCreating final dataset for matched facades...")
    combined_rows = []

    if not matched_facades_gdf.empty:
        for idx, facade_row in matched_facades_gdf.iterrows():
            camera_idx = facade_row['matched_camera_idx']
            camera_row = gdf_camera_subset.loc[camera_idx]
            
            building_matches = gdf_projected[gdf_projected['osm_id'] == facade_row['building_id']]
            if building_matches.empty:
                print(f"Warning: Building ID {facade_row['building_id']} not found in gdf_projected. Skipping facade.")
                continue
            building_info = building_matches.iloc[0]
            
            combined_row = {
                'building_id': facade_row['building_id'],
                'osm_id': facade_row['building_id'],
                'group_id': building_info['group_id'],
                'geometry_midpoint': facade_row['geometry_midpoint'],
                'geometry_outer_edge': facade_row['geometry_segment'],
                'orientation': facade_row['orientation'],
                'facade_length': facade_row['length'],
                'distance_to_camera': facade_row['camera_distance'],
                'selection_score': facade_row.get('selection_score', 0),
                'segment_count': facade_row.get('segment_count', 1),
                'camera_idx': camera_idx,
                'X': camera_row.get('X', camera_row.geometry.x if hasattr(camera_row.geometry, 'x') else None),
                'Y': camera_row.get('Y', camera_row.geometry.y if hasattr(camera_row.geometry, 'y') else None),
                'Z': camera_row.get('Z', camera_row.get('z', 0)), 
                'PATH': camera_row.get('PATH', ''),
                'RUN': camera_row.get('RUN', ''),
                'FOTO': camera_row.get('FOTO', ''),
                'orientation_run': camera_row.get('orientation_run', None),
                'Yaw': camera_row.get('Yaw', camera_row.get('yaw', None)),
                'Pitch': camera_row.get('Pitch', camera_row.get('pitch', None)),
                'Roll': camera_row.get('Roll', camera_row.get('roll', None)),
                'viewing_quality': facade_row.get('viewing_quality', None) 
            }
            combined_rows.append(combined_row)

    expected_columns_final = [
        'building_id', 'osm_id', 'group_id', 'geometry_midpoint', 'geometry_outer_edge',
        'orientation', 'facade_length', 'distance_to_camera', 'selection_score',
        'segment_count', 'camera_idx', 'X', 'Y', 'Z', 'PATH', 'RUN', 'FOTO',
        'Yaw', 'Pitch', 'Roll', 'viewing_quality'
    ]

    if not combined_rows:
        print("No facades were matched to create gdf_combined. Creating an empty GeoDataFrame.")
        temp_df = pd.DataFrame(columns=[col for col in expected_columns_final if col != 'geometry_midpoint'])
        geometry_col = gpd.GeoSeries([], name='geometry_midpoint', crs=TARGET_CRS)
        gdf_combined = gpd.GeoDataFrame(temp_df, geometry=geometry_col, crs=TARGET_CRS)
        for col in expected_columns_final:
            if col not in gdf_combined.columns and col != 'geometry_midpoint':
                gdf_combined[col] = pd.NA
    else:
        gdf_combined = gpd.GeoDataFrame(combined_rows, geometry='geometry_midpoint', crs=TARGET_CRS)

  
    # Calculate yaw angles from camera to building features
    print("\nCalculating yaw angles...")
    if not gdf_combined.empty:
        for idx, row in gdf_combined.iterrows():
            cam_x = row['X'] if pd.notna(row['X']) else 0
            cam_y = row['Y'] if pd.notna(row['Y']) else 0
            
            midpoint = row['geometry_midpoint']
            if midpoint:
                gdf_combined.at[idx, 'midpoint_yaw_rad'] = calculate_yaw_radians(cam_x, cam_y, midpoint.x, midpoint.y)
            
            edge = row['geometry_outer_edge']
            if edge and not edge.is_empty:
                coords = list(edge.coords)
                if coords:
                    gdf_combined.at[idx, 'edge_first_yaw_rad'] = calculate_yaw_radians(cam_x, cam_y, coords[0][0], coords[0][1])
                    if len(coords) > 1:
                        gdf_combined.at[idx, 'edge_second_yaw_rad'] = calculate_yaw_radians(cam_x, cam_y, coords[-1][0], coords[-1][1])
                    else:
                        gdf_combined.at[idx, 'edge_second_yaw_rad'] = None
                else:
                    gdf_combined.at[idx, 'edge_first_yaw_rad'] = None
                    gdf_combined.at[idx, 'edge_second_yaw_rad'] = None
            else:
                gdf_combined.at[idx, 'edge_first_yaw_rad'] = None
                gdf_combined.at[idx, 'edge_second_yaw_rad'] = None
        print("Yaw calculations complete")

        gdf_combined['midpoint_x'] = gdf_combined['geometry_midpoint'].apply(lambda p: p.x if p else None)
        gdf_combined['midpoint_y'] = gdf_combined['geometry_midpoint'].apply(lambda p: p.y if p else None)

        output_filename = f"Enhanced_facades_for_extraction_{len(gdf_combined)}_facades.geojson"
        output_path = os.path.join(output_dir, output_filename)
        gdf_combined.to_file(output_path, driver="GeoJSON")
        print(f"\nSaved enhanced facade data to: {output_path}")

        if not gdf_combined.empty:
            summary_cols = ['building_id', 'orientation', 'facade_length', 'distance_to_camera', 
                        'X', 'Y', 'RUN', 'FOTO', 'viewing_quality']
            available_cols = [col for col in summary_cols if col in gdf_combined.columns]
            summary_df = gdf_combined[available_cols].copy()
            summary_df['midpoint_x'] = gdf_combined['midpoint_x'] if 'midpoint_x' in gdf_combined.columns else None
            summary_df['midpoint_y'] = gdf_combined['midpoint_y'] if 'midpoint_y' in gdf_combined.columns else None
            
            summary_filename = f"Enhanced_facades_summary_{len(gdf_combined)}_facades.csv"
            summary_path = os.path.join(output_dir, summary_filename)
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary CSV to: {summary_path}")
    else:
        print("gdf_combined is empty. Skipping yaw calculations and saving.")



    # Enhanced Visualization with multiple plots - Support both interactive and file output
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Facade Extraction and Matching Visualization", fontsize=16)

    # ==== Plot 1: Facades with midpoints (showing matched vs unmatched) ====
    ax1 = axes[0, 0]
    if not gdf_projected.empty:
        gdf_projected.plot(ax=ax1, color='lightgrey', edgecolor='darkgrey', alpha=0.7, linewidth=0.5, label='Buildings')

    if 'geometry_segment' in matched_facades_gdf.columns and not matched_facades_gdf.empty :
        matched_facades_gdf.plot(ax=ax1, color='blue', linewidth=2, label=f'Matched ({len(matched_facades_gdf)})')
        if 'geometry_midpoint' in matched_facades_gdf.columns:
            gpd.GeoDataFrame(matched_facades_gdf, geometry='geometry_midpoint', crs=matched_facades_gdf.crs).plot(ax=ax1, color='black', markersize=15, marker='o')

    if 'geometry_segment' in unmatched_facades_gdf.columns and not unmatched_facades_gdf.empty:
        unmatched_facades_gdf.plot(ax=ax1, color='lightcoral', linewidth=1, alpha=0.7, label=f'Unmatched ({len(unmatched_facades_gdf)})')
        if 'geometry_midpoint' in unmatched_facades_gdf.columns:
            gpd.GeoDataFrame(unmatched_facades_gdf, geometry='geometry_midpoint', crs=unmatched_facades_gdf.crs).plot(ax=ax1, color='darkred', markersize=10, marker='x')

    ax1.legend(loc='upper left')
    ax1.set_title("Facades (Matched vs. Unmatched)", fontsize=14)
    ax1.set_xlabel("X Coordinate"); ax1.set_ylabel("Y Coordinate")

    # ==== Plot 2: Camera positions (all available vs actually used) ====
    ax2 = axes[0, 1]
    if not gdf_projected.empty:
        gdf_projected.plot(ax=ax2, color='lightgrey', edgecolor='darkgrey', alpha=0.7, linewidth=0.5, label='Buildings')

    if not gdf_camera_subset.empty:
        gdf_camera_subset.plot(ax=ax2, color='lightblue', markersize=20, marker='^', alpha=0.6, label=f'All Cameras ({len(gdf_camera_subset)})')

    used_cameras = gpd.GeoDataFrame() # Initialize as empty
    if not matched_facades_gdf.empty and 'matched_camera_idx' in matched_facades_gdf.columns:
        used_camera_indices = matched_facades_gdf['matched_camera_idx'].unique()
        if len(used_camera_indices) > 0:
            used_cameras = gdf_camera_subset.loc[used_camera_indices]
            used_cameras.plot(ax=ax2, color='navy', markersize=30, marker='o', label=f'Used Cameras ({len(used_cameras)})')

    ax2.legend(loc='upper left')
    ax2.set_title("Camera Positions (All vs. Used)", fontsize=14)
    ax2.set_xlabel("X Coordinate"); ax2.set_ylabel("Y Coordinate")

    # ==== Plot 3: Facade-Camera connections showing viewing relationships ====
    ax3 = axes[1, 0]
    if not gdf_projected.empty:
        gdf_projected.plot(ax=ax3, color='lightgrey', edgecolor='darkgrey', alpha=0.7, linewidth=0.5, label='Buildings')

    if not matched_facades_gdf.empty and 'camera_geometry' in matched_facades_gdf.columns and 'geometry_midpoint' in matched_facades_gdf.columns:
        for idx, facade in matched_facades_gdf.iterrows():
            midpoint = facade['geometry_midpoint']
            camera = facade['camera_geometry']
            if midpoint and camera: # Ensure both geometries exist
                ax3.plot([camera.x, midpoint.x], [camera.y, midpoint.y], 'b-', alpha=0.2, linewidth=0.5)
        
        if 'geometry_segment' in matched_facades_gdf.columns:
            matched_facades_gdf.plot(ax=ax3, color='green', linewidth=2, label='Matched Facades')
        if not used_cameras.empty:
            used_cameras.plot(ax=ax3, color='blue', markersize=30, marker='o', label='Used Cameras')
    elif not matched_facades_gdf.empty and 'geometry_segment' in matched_facades_gdf.columns: # Plot facades even if no connections
        matched_facades_gdf.plot(ax=ax3, color='green', linewidth=2, label='Matched Facades')

    ax3.legend(loc='upper left')
    ax3.set_title("Facade-Camera Viewing Connections", fontsize=14)
    ax3.set_xlabel("X Coordinate"); ax3.set_ylabel("Y Coordinate")

    # ==== Plot 4: Final result - Matched Facades by orientation with midpoints ====
    ax4 = axes[1, 1]
    if not gdf_projected.empty:
        gdf_projected.plot(ax=ax4, color='lightgrey', edgecolor='darkgrey', alpha=0.7, linewidth=0.5, label='Buildings')

    colors = {'N': 'blue', 'E': 'green', 'S': 'red', 'W': 'purple', 'nan': 'grey'} # Added nan for safety

    if not gdf_combined.empty and 'geometry_outer_edge' in gdf_combined.columns:
        # Convert orientation to string to handle potential NaNs in groupby
        gdf_combined['orientation_str'] = gdf_combined['orientation'].astype(str)

        for orientation_val, group in gdf_combined.groupby('orientation_str'):
            color = colors.get(orientation_val, 'grey') # Use 'grey' for unexpected orientations
            # Plot facade lines
            gpd.GeoDataFrame(group, geometry='geometry_outer_edge', crs=gdf_combined.crs).plot(
                ax=ax4, color=color, linewidth=2, label=f'{orientation_val} ({len(group)})'
            )
            # Plot midpoints
            if 'geometry_midpoint' in group.columns:
                gpd.GeoDataFrame(group, geometry='geometry_midpoint', crs=gdf_combined.crs).plot(
                    ax=ax4, color=color, markersize=15, marker='o', edgecolor='black', linewidth=0.5
                )
        # Drop the temporary column
        gdf_combined.drop(columns=['orientation_str'], inplace=True, errors='ignore')

    if not used_cameras.empty: # Plot used cameras from previous plot
        used_cameras.plot(ax=ax4, color='black', markersize=25, marker='s', label='Used Cameras')

    handles, labels = ax4.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels for legend
    ax4.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')
    ax4.set_title("Final Matched Facades by Orientation", fontsize=14)
    ax4.set_xlabel("X Coordinate"); ax4.set_ylabel("Y Coordinate")

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    # Handle plotting based on mode
    if INTERACTIVE_PLOTS:
        print("Displaying interactive plots...")
        plt.show()  # Show interactive plots
        
        # Also save to file for reference
        plot_filename = f"facade_extraction_visualization_{len(gdf_combined)}_facades.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Also saved plot to: {plot_path}")
    else:
        # Save the plot to file only
        plot_filename = f"facade_extraction_visualization_{len(gdf_combined)}_facades.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization plot to: {plot_path}")
        plt.close()  # Close the figure to free memory

    # Create a simple summary plot showing the study area
    if not gdf_camera_subset.empty and not gdf_projected.empty:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot buildings
        gdf_projected.plot(ax=ax, color='lightgrey', edgecolor='darkgrey', alpha=0.7, linewidth=0.5, label='Buildings')
        
        # Plot all cameras
        gdf_camera_subset.plot(ax=ax, color='blue', markersize=30, marker='^', alpha=0.8, label=f'Camera Points ({len(gdf_camera_subset)})')
        
        # Plot matched facades if any
        if not gdf_combined.empty and 'geometry_outer_edge' in gdf_combined.columns:
            gpd.GeoDataFrame(gdf_combined, geometry='geometry_outer_edge', crs=gdf_combined.crs).plot(
                ax=ax, color='red', linewidth=3, alpha=0.8, label=f'Matched Facades ({len(gdf_combined)})'
            )
        
        ax.legend(loc='upper left')
        ax.set_title(f"Study Area Overview\nDynamic Bbox: {lat_min:.4f}, {lon_min:.4f} to {lat_max:.4f}, {lon_max:.4f}\nUsing {len(RUN_VALUES)} RUN values", fontsize=14)
        ax.set_xlabel("X Coordinate (UTM)")
        ax.set_ylabel("Y Coordinate (UTM)")
        
        if INTERACTIVE_PLOTS:
            print("Displaying overview plot...")
            plt.show()
            
            # Also save overview plot
            overview_filename = f"study_area_overview_{len(gdf_combined)}_facades.png"
            overview_path = os.path.join(output_dir, overview_filename)
            plt.savefig(overview_path, dpi=300, bbox_inches='tight')
            print(f"Also saved overview plot to: {overview_path}")
        else:
            # Save overview plot
            overview_filename = f"study_area_overview_{len(gdf_combined)}_facades.png"
            overview_path = os.path.join(output_dir, overview_filename)
            plt.savefig(overview_path, dpi=300, bbox_inches='tight')
            print(f"Saved overview plot to: {overview_path}")
            plt.close()


    # Print detailed matching statistics
    print("\n" + "="*50)
    print("MATCHING STATISTICS")
    print("="*50)
    print(f"Total facades available for matching (after filtering): {len(best_facades_gdf)}")
    print(f"Facades with suitable camera: {len(matched_facades_gdf)}")
    print(f"Facades without suitable camera: {len(unmatched_facades_gdf)}")
    if not matched_facades_gdf.empty and 'camera_distance' in matched_facades_gdf.columns:
        print(f"\nAverage distance to camera: {matched_facades_gdf['camera_distance'].mean():.2f} meters")
        print(f"Max distance to camera: {matched_facades_gdf['camera_distance'].max():.2f} meters")
        print(f"Min distance to camera: {matched_facades_gdf['camera_distance'].min():.2f} meters")

    print("\nMatched facades by orientation:")
    if not gdf_combined.empty and 'orientation' in gdf_combined.columns:
        print(gdf_combined['orientation'].value_counts())
    else:
        print("No matched facades to show orientation for.")

    print("\nUnmatched facades by orientation:")
    if not unmatched_facades_gdf.empty and 'orientation' in unmatched_facades_gdf.columns:
        print(unmatched_facades_gdf['orientation'].value_counts())
    else:
        print("No unmatched facades to show orientation for.")

    if not matched_facades_gdf.empty and 'matched_camera_idx' in matched_facades_gdf.columns:
        camera_usage_stats = matched_facades_gdf['matched_camera_idx'].value_counts()
        if not camera_usage_stats.empty:
            print(f"\nCamera usage (facades per camera):")
            print(f"  Average facades per camera: {camera_usage_stats.mean():.2f}")
            print(f"  Max facades per camera: {camera_usage_stats.max()}")
            print(f"  Cameras used: {len(camera_usage_stats)} out of {len(gdf_camera_subset)}")
        else:
            print("\nNo cameras were used for matching.")
    else:
        print("\nNo camera usage statistics available (no matched facades or missing 'matched_camera_idx').")



    # Print summary statistics
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total buildings from OSM: {len(buildings)}")
    print(f"Buildings after initial filtering (valid geometry, area): {len(gdf_projected)}")
    if not gdf_projected.empty and 'group_id' in gdf_projected.columns:
        print(f"Building groups formed: {gdf_projected['group_id'].nunique()}")
    else:
        print("Building groups formed: 0 (or gdf_projected is empty)")
    print(f"Exterior facade segments extracted: {len(facade_segments_gdf)}")
    print(f"Street-facing facades (initial filter): {len(facade_segments_filtered)}")
    print(f"Facades after merging adjacent segments: {len(facade_segments_merged)}")
    print(f"Total facades available for final matching: {len(best_facades_gdf)}")
    print(f"Facades finally matched with cameras: {len(matched_facades_gdf)}")
    print(f"Facades discarded (e.g. no suitable camera, LoS issues): {len(unmatched_facades_gdf)}")

    if not gdf_combined.empty:
        if 'distance_to_camera' in gdf_combined.columns:
            print(f"\nAverage distance to camera (final matched): {gdf_combined['distance_to_camera'].mean():.2f} meters")
        if 'facade_length' in gdf_combined.columns:
            print(f"Average facade length (final matched): {gdf_combined['facade_length'].mean():.2f} meters")
        if 'orientation' in gdf_combined.columns:
            print("\nFinal matched facades by orientation:")
            print(gdf_combined['orientation'].value_counts())
        if 'group_id' in gdf_combined.columns:
            print("\nFinal matched facades by group type:")
            standalone_count = len(gdf_combined[gdf_combined['group_id'].astype(str).str.startswith('standalone')])
            block_count = len(gdf_combined[gdf_combined['group_id'].astype(str).str.startswith('block')])
            print(f"  Standalone buildings: {standalone_count}")
            print(f"  Buildings in blocks: {block_count}")
    else:
        print("\nNo final matched facades to summarize.")

    # Show what columns we have in the final dataset
    print("\nColumns in final dataset (gdf_combined):")
    if not gdf_combined.empty:
        print(gdf_combined.columns.tolist())
    else:
        print("gdf_combined is empty.")