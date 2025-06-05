from geodataframe.utils.libraries import *

"""
# These functions improve facade extraction by filtering street-facing facades and handling complex cases.
"""

def is_facade_facing_camera(facade_segment, camera_point, building_polygon=None):
    """Check if a facade segment is facing towards the camera point using outward-facing normal vector."""
    coords = list(facade_segment.coords)
    if len(coords) < 2:
        return False
    
    p1 = np.array(coords[0])
    p2 = np.array(coords[-1])
    
    facade_vec = p2 - p1
    if np.linalg.norm(facade_vec) < 1e-9:
        return False
    
    normal_1 = np.array([-facade_vec[1], facade_vec[0]])
    normal_2 = np.array([facade_vec[1], -facade_vec[0]])
    
    midpoint = facade_segment.centroid
    to_camera = np.array([camera_point.x - midpoint.x, camera_point.y - midpoint.y])
    
    if np.linalg.norm(to_camera) < 1e-9:
        return True
    
    to_camera_norm = to_camera / np.linalg.norm(to_camera)
    
    if building_polygon is not None:
        building_centroid = building_polygon.centroid
        to_facade = np.array([midpoint.x - building_centroid.x, midpoint.y - building_centroid.y])
        
        if np.linalg.norm(to_facade) > 1e-9:
            alignment_1 = np.dot(normal_1, to_facade)
            alignment_2 = np.dot(normal_2, to_facade)
            
            if alignment_1 > alignment_2:
                outward_normal = normal_1
            else:
                outward_normal = normal_2
                
            outward_normal = outward_normal / (np.linalg.norm(outward_normal) + 1e-10)
            return np.dot(outward_normal, to_camera_norm) > 0
    
    alignment_1 = np.dot(normal_1, to_camera_norm)
    alignment_2 = np.dot(normal_2, to_camera_norm)
    
    return max(alignment_1, alignment_2) > 0

def filter_street_facing_facades(facade_segments_gdf, camera_points_gdf, buildings_gdf=None, max_distance=30):
    """Filter facades to keep only those facing streets and within reasonable distance."""
    print(f"Filtering {len(facade_segments_gdf)} facades for street-facing orientation...")
    
    street_facing = []
    if facade_segments_gdf.empty:
        print("No facades to filter.")
        return gpd.GeoDataFrame(street_facing, geometry='geometry_segment', crs=facade_segments_gdf.crs if facade_segments_gdf.crs else "EPSG:4326")

    building_lookup = {}
    if buildings_gdf is not None:
        building_lookup = {row['osm_id']: row['geometry'] for _, row in buildings_gdf.iterrows()}
        print(f"Using {len(building_lookup)} building polygons for outward normal calculation")

    for idx, facade in tqdm(facade_segments_gdf.iterrows(), total=len(facade_segments_gdf)):
        distances = camera_points_gdf.geometry.distance(facade.geometry_midpoint)
        min_distance = distances.min()
        
        if min_distance <= max_distance:
            nearest_camera_idx = distances.idxmin()
            nearest_camera = camera_points_gdf.loc[nearest_camera_idx]
            
            building_polygon = None
            if hasattr(facade, 'building_id') and facade.building_id in building_lookup:
                building_polygon = building_lookup[facade.building_id]
            
            if is_facade_facing_camera(facade.geometry_segment, nearest_camera.geometry, building_polygon):
                facade_dict = facade.to_dict()
                facade_dict['distance_to_street'] = min_distance
                facade_dict['nearest_camera_idx'] = nearest_camera_idx
                street_facing.append(facade_dict)
    
    result_gdf = gpd.GeoDataFrame(street_facing, geometry='geometry_segment', crs=facade_segments_gdf.crs)
    print(f"Found {len(result_gdf)} street-facing facades")
    return result_gdf

def segments_are_adjacent(seg1, seg2, distance_tolerance=2):
    """Check if two segments are adjacent (close enough to be part of same facade)."""
    endpoints = [
        (Point(seg1.coords[0]), Point(seg2.coords[0])),
        (Point(seg1.coords[0]), Point(seg2.coords[-1])),
        (Point(seg1.coords[-1]), Point(seg2.coords[0])),
        (Point(seg1.coords[-1]), Point(seg2.coords[-1]))
    ]
    
    for p1, p2 in endpoints:
        if p1.distance(p2) < distance_tolerance:
            return True
    return False

def merge_adjacent_facade_segments(facade_segments_gdf, angle_tolerance=15, distance_tolerance=2):
    """Merge facade segments that are likely part of the same logical facade."""
    print("Merging adjacent facade segments...")
    if facade_segments_gdf.empty:
        print("No facades to merge.")
        return facade_segments_gdf.copy()

    merged_facades = []
    processed = set()
    
    for idx, facade in facade_segments_gdf.iterrows():
        if idx in processed:
            continue
        
        candidates = []
        for other_idx, other in facade_segments_gdf.iterrows():
            if other_idx != idx and other_idx not in processed:
                if (facade.building_id == other.building_id and 
                    facade.orientation == other.orientation):
                    if segments_are_adjacent(facade.geometry_segment, other.geometry_segment, distance_tolerance):
                        candidates.append(other_idx)
        
        if candidates:
            segments_to_merge = [facade.geometry_segment] + [facade_segments_gdf.loc[i].geometry_segment for i in candidates]
            
            try:
                merged_line = linemerge(segments_to_merge)
                if isinstance(merged_line, LineString):
                    merged_facade = facade.to_dict()
                    merged_facade['geometry_segment'] = merged_line
                    merged_facade['geometry_midpoint'] = merged_line.centroid
                    merged_facade['length'] = merged_line.length
                    merged_facade['segment_count'] = len(candidates) + 1
                    merged_facades.append(merged_facade)
                else:
                    merged_facades.append(facade.to_dict())
            except:
                merged_facades.append(facade.to_dict())
            
            processed.add(idx)
            processed.update(candidates)
        else:
            merged_facades.append(facade.to_dict())
            processed.add(idx)
    
    print(f"Merged to {len(merged_facades)} facades")
    if not merged_facades:
        return gpd.GeoDataFrame(geometry=[], crs=facade_segments_gdf.crs)
        
    return gpd.GeoDataFrame(merged_facades, geometry='geometry_segment', crs=facade_segments_gdf.crs)

def calculate_viewing_angle_quality(facade_segment, camera_point):
    """Calculate the quality of viewing angle (0-1, where 1 is perpendicular)."""
    midpoint = facade_segment.centroid
    view_vec = np.array([midpoint.x - camera_point.x, midpoint.y - camera_point.y])
    if np.linalg.norm(view_vec) == 0: 
        return 0
    view_vec = view_vec / np.linalg.norm(view_vec)
    
    coords = list(facade_segment.coords)
    facade_vec = np.array([coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]])
    if np.linalg.norm(facade_vec) == 0: 
        return 0
    facade_vec = facade_vec / np.linalg.norm(facade_vec)
    
    dot_product = abs(np.dot(view_vec, facade_vec))
    quality = 1 - dot_product
    return quality

def select_best_facade_for_building(building_facades, camera_points_gdf):
    """Select the best facade for a building based on multiple criteria."""
    if building_facades.empty: 
        return pd.Series()
    if len(building_facades) == 1:
        return building_facades.iloc[0]
    
    scores = []
    for idx, facade in building_facades.iterrows():
        score = 0
        if 'distance_to_street' in facade:
            score += 100 / (1 + facade.distance_to_street)
        else:
            min_distance = camera_points_gdf.geometry.distance(facade.geometry_midpoint).min()
            score += 100 / (1 + min_distance)
        
        score += facade.length * 10
        
        if 'nearest_camera_idx' in facade and facade.nearest_camera_idx in camera_points_gdf.index:
            camera = camera_points_gdf.loc[facade.nearest_camera_idx]
        else:
            distances = camera_points_gdf.geometry.distance(facade.geometry_midpoint)
            if distances.empty:
                camera = None
            else:
                camera = camera_points_gdf.loc[distances.idxmin()]
        
        if camera is not None:
             angle_quality = calculate_viewing_angle_quality(facade.geometry_segment, camera.geometry)
             score += angle_quality * 50
        
        if 'segment_count' in facade:
            score += facade.segment_count * 5
        scores.append(score)
    
    if not scores: 
        return pd.Series()

    best_idx_num = np.argmax(scores)
    best_facade_series = building_facades.iloc[best_idx_num].copy()
    best_facade_series['selection_score'] = scores[best_idx_num]
    return best_facade_series
