from geodataframe.utils.libraries import *

"""
These functions handle camera orientation and facade orientation calculations.
"""

def calculate_yaw_radians(cam_x, cam_y, tgt_x, tgt_y):
    return math.atan2(tgt_y - cam_y, tgt_x - cam_x)

def calculate_yaw_degrees(camera_coords, target_coords):
    x1, y1 = camera_coords[:2]
    x2, y2 = target_coords[:2]
    dx = x2 - x1
    dy = y2 - y1
    yaw = np.arctan2(-dy, -dx)
    yaw_deg = np.degrees(yaw)
    yaw_deg = (yaw_deg + 360) % 360
    return yaw_deg

def calculate_azimuth(p1, p2):
    """Calculate azimuth (bearing) from point p1 to point p2. Returns azimuth in degrees (0-360) where 0° is North."""
    lon1, lat1 = p1
    lon2, lat2 = p2
    
    dx = lon2 - lon1
    dy = lat2 - lat1
    
    azimuth_rad = math.atan2(dx, dy)
    azimuth_deg = math.degrees(azimuth_rad)
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg

def classify_orientation(azimuth):
    """Classify azimuth into cardinal directions (N, E, S, W) using 45-degree sectors."""
    azimuth = azimuth % 360
    
    if azimuth >= 315 or azimuth < 45:
        return 'N'
    elif 45 <= azimuth < 135:
        return 'E'
    elif 135 <= azimuth < 225:
        return 'S'
    elif 225 <= azimuth < 315:
        return 'W'
    else:
        return 'N'

def calculate_outward_facing_normal_orientation(segment, building_polygon):
    """Calculate the outward-facing normal vector orientation for a facade segment."""
    coords = list(segment.coords)
    if len(coords) < 2:
        return 0.0
    
    p1 = np.array(coords[0])
    p2 = np.array(coords[-1])
    segment_vector = p2 - p1
    
    if np.linalg.norm(segment_vector) < 1e-9:
        return 0.0
    
    normal_1 = np.array([-segment_vector[1], segment_vector[0]])
    normal_2 = np.array([segment_vector[1], -segment_vector[0]])
    
    segment_midpoint = segment.centroid
    building_centroid = building_polygon.centroid
    
    to_facade = np.array([segment_midpoint.x - building_centroid.x, 
                         segment_midpoint.y - building_centroid.y])
    
    if np.linalg.norm(to_facade) < 1e-9:
        outward_normal = normal_1
    else:
        alignment_1 = np.dot(normal_1, to_facade)
        alignment_2 = np.dot(normal_2, to_facade)
        
        if alignment_1 > alignment_2:
            outward_normal = normal_1
        else:
            outward_normal = normal_2
    
    azimuth_rad = math.atan2(outward_normal[0], outward_normal[1])
    azimuth_deg = math.degrees(azimuth_rad)
    azimuth_deg = (azimuth_deg + 360) % 360
    
    return azimuth_deg

def add_orientation_to_facades(facade_segments_gdf, buildings_gdf):
    """Add orientation labels to facade segments using outward-facing normal vectors."""
    print("Calculating facade orientations using outward-facing normal vectors...")
    
    if facade_segments_gdf.empty:
        facade_segments_gdf['orientation'] = []
        facade_segments_gdf['azimuth'] = []
        return facade_segments_gdf
    
    orientations = []
    azimuths = []
    
    building_lookup = {row['osm_id']: row['geometry'] for _, row in buildings_gdf.iterrows()}
    
    for idx, row in facade_segments_gdf.iterrows():
        segment = row.geometry_segment
        building_id = row.building_id
        
        if building_id in building_lookup:
            building_polygon = building_lookup[building_id]
            azimuth = calculate_outward_facing_normal_orientation(segment, building_polygon)
            orientation = classify_orientation(azimuth)
            
            orientations.append(orientation)
            azimuths.append(azimuth)
        else:
            orientations.append('N')
            azimuths.append(0.0)
            print(f"Warning: Building {building_id} not found for facade segment {idx}")
    
    facade_segments_gdf['orientation'] = orientations
    facade_segments_gdf['azimuth'] = azimuths
    
    print(f"Orientation calculation complete. Distribution:")
    if orientations:
        orientation_counts = pd.Series(orientations).value_counts()
        for orient, count in orientation_counts.items():
            print(f"  {orient}: {count}")
    
    return facade_segments_gdf