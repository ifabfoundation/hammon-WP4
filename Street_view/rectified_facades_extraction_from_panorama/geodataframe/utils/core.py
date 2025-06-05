from geodataframe.utils.libraries import *

"""
These functions handle context-aware facade selection for urban vs suburban areas
"""

def analyze_building_density_distribution(facades_gdf):
    """Analyze the distribution of building group sizes to help choose optimal threshold."""
    if facades_gdf.empty or 'group_id' not in facades_gdf.columns:
        print("No building groups to analyze")
        return
    
    buildings_per_group = facades_gdf.groupby('group_id')['building_id'].nunique()
    
    print(f"\nBuilding density analysis:")
    print(f"  Total groups: {len(buildings_per_group)}")
    print(f"  Group size distribution:")
    
    for size in sorted(buildings_per_group.unique()):
        count = (buildings_per_group == size).sum()
        percentage = count / len(buildings_per_group) * 100
        print(f"    {size} building(s): {count} groups ({percentage:.1f}%)")
    
    print(f"  Average buildings per group: {buildings_per_group.mean():.1f}")
    print(f"  Median buildings per group: {buildings_per_group.median():.1f}")
    
    standalone_groups = (buildings_per_group == 1).sum()
    small_groups = (buildings_per_group <= 3).sum()
    total_groups = len(buildings_per_group)
    
    if standalone_groups / total_groups > 0.5:
        suggested_threshold = 4
        print(f"  Suggested threshold: {suggested_threshold} (many standalone buildings detected)")
    elif small_groups / total_groups > 0.7:
        suggested_threshold = 5
        print(f"  Suggested threshold: {suggested_threshold} (suburban/mixed area detected)")
    else:
        suggested_threshold = 6
        print(f"  Suggested threshold: {suggested_threshold} (urban area detected)")
    
    return suggested_threshold

def select_best_facade_considering_cameras(building_facades, camera_points_gdf):
    """Enhanced facade selection that considers camera positions and viewing quality."""
    if building_facades.empty:
        return pd.Series()
    
    if len(building_facades) == 1:
        return building_facades.iloc[0]
    
    facade_scores = []
    
    for idx, facade in building_facades.iterrows():
        facade_midpoint = facade['geometry_midpoint']
        facade_segment = facade['geometry_segment']
        facade_orientation = facade['orientation']
        
        distances = camera_points_gdf.geometry.distance(facade_midpoint)
        min_distance = distances.min()
        nearest_camera_idx = distances.idxmin()
        nearest_camera = camera_points_gdf.loc[nearest_camera_idx]
        
        distance_score = 1000 * np.exp(-min_distance / 10)
        length_bonus = facade['length'] * 2
        score = distance_score + length_bonus
        
        to_camera = np.array([nearest_camera.geometry.x - facade_midpoint.x, 
                            nearest_camera.geometry.y - facade_midpoint.y])
        if np.linalg.norm(to_camera) > 1e-9:
            to_camera_norm = to_camera / np.linalg.norm(to_camera)
            
            coords = list(facade_segment.coords)
            p1 = np.array(coords[0])
            p2 = np.array(coords[-1])
            segment_vector = p2 - p1
            
            if np.linalg.norm(segment_vector) > 1e-9:
                normal_1 = np.array([-segment_vector[1], segment_vector[0]])
                normal_2 = np.array([segment_vector[1], -segment_vector[0]])
                
                alignment_1 = np.dot(normal_1, to_camera_norm)
                alignment_2 = np.dot(normal_2, to_camera_norm)
                
                if alignment_1 > alignment_2:
                    facade_normal = normal_1
                    alignment = alignment_1
                else:
                    facade_normal = normal_2
                    alignment = alignment_2
                
                if alignment > 0.7:
                    score += 30
                elif alignment > 0.3:
                    score += 15
                elif alignment < -0.3:
                    score -= 30
        
        facade_scores.append((score, idx))
    
    facade_scores.sort(reverse=True)
    best_idx = facade_scores[0][1]
    best_facade = building_facades.loc[best_idx].copy()
    best_facade['selection_score'] = facade_scores[0][0]
    
    return best_facade

def smart_facade_selection_by_context(facades_gdf, camera_points_gdf, building_density_threshold=5):
    """Enhanced smart facade selection with better camera-aware selection."""
    print(f"Applying enhanced smart facade selection (density threshold: {building_density_threshold} buildings)...")
    
    if facades_gdf.empty:
        return facades_gdf.copy()
    
    selected_facades = []
    
    for building_id, building_facades in facades_gdf.groupby('building_id'):
        
        if len(building_facades) == 1:
            selected_facade = building_facades.iloc[0].copy()
            selected_facade['selection_reason'] = 'single_facade'
            selected_facades.append(selected_facade)
            continue
        
        group_id = building_facades.iloc[0]['group_id']
        buildings_in_group = facades_gdf[facades_gdf['group_id'] == group_id]['building_id'].nunique()
        
        orientations = building_facades['orientation'].unique()
        is_corner_building = len(orientations) > 1
        
        if buildings_in_group < building_density_threshold:
            best_facade = select_best_facade_considering_cameras(
                building_facades, camera_points_gdf
            )
            
            if not best_facade.empty:
                best_facade['selection_reason'] = 'suburban_camera_aware_best'
                best_facade['buildings_in_group'] = buildings_in_group
                selected_facades.append(best_facade)
                
        else:
            if is_corner_building and len(orientations) <= 2:
                for orientation in orientations:
                    orientation_facades = building_facades[building_facades['orientation'] == orientation]
                    
                    if len(orientation_facades) == 1:
                        selected_facade = orientation_facades.iloc[0].copy()
                        selected_facade['selection_reason'] = f'urban_corner_{orientation}'
                    else:
                        selected_facade = select_best_facade_considering_cameras(
                            orientation_facades, camera_points_gdf
                        )
                        if not selected_facade.empty:
                            selected_facade['selection_reason'] = f'urban_corner_best_{orientation}'
                    
                    if not selected_facade.empty:
                        selected_facade['buildings_in_group'] = buildings_in_group
                        selected_facades.append(selected_facade)
            else:
                best_facade = select_best_facade_considering_cameras(
                    building_facades, camera_points_gdf
                )
                
                if not best_facade.empty:
                    best_facade['selection_reason'] = 'urban_complex_best'
                    best_facade['buildings_in_group'] = buildings_in_group
                    selected_facades.append(best_facade)
    
    if not selected_facades:
        return gpd.GeoDataFrame(columns=facades_gdf.columns, geometry='geometry_segment', crs=facades_gdf.crs)
    
    result_gdf = gpd.GeoDataFrame(selected_facades, geometry='geometry_segment', crs=facades_gdf.crs)
    
    if 'selection_reason' in result_gdf.columns and not result_gdf.empty:
        result_gdf['selection_reason'] = result_gdf['selection_reason'].fillna('')
        
        suburban_count = len(result_gdf[result_gdf['selection_reason'].str.contains('suburban', na=False)])
        urban_count = len(result_gdf[result_gdf['selection_reason'].str.contains('urban', na=False)])
        single_count = len(result_gdf[result_gdf['selection_reason'].str.contains('single', na=False)])
    else:
        suburban_count = urban_count = single_count = 0
    
    print(f"Enhanced selection results:")
    print(f"  Single facade buildings: {single_count}")
    print(f"  Suburban mode (< {building_density_threshold} buildings): {suburban_count} facades")
    print(f"    - Strategy: Camera-aware best facade selection")
    print(f"  Urban mode (≥ {building_density_threshold} buildings): {urban_count} facades")
    print(f"    - Strategy: Selective corner building handling")
    print(f"  Total selected: {len(result_gdf)} facades")
    
    if 'buildings_in_group' in result_gdf.columns and not result_gdf.empty:
        print(f"  Average buildings per group: {result_gdf['buildings_in_group'].mean():.1f}")
    
    if 'selection_reason' in result_gdf.columns and not result_gdf.empty:
        print(f"  Selection reasons:")
        reason_counts = result_gdf['selection_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count}")
    
    return result_gdf