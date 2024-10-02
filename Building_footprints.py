import overpy
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, linemerge
from shapely.validation import make_valid
from shapely.errors import TopologicalError
import matplotlib.pyplot as plt
import networkx as nx
import contextily as cx
import plotly.graph_objects as go

# Helper function to close small gaps between lines
def close_small_gaps(line, tolerance=0.0001):
    if line.is_ring:  # Already closed
        return line
    # Check if the start and end points are within the tolerance
    if line.coords[0] != line.coords[-1]:
        start = line.coords[0]
        end = line.coords[-1]
        if LineString([start, end]).length <= tolerance:
            coords = list(line.coords)
            coords[-1] = start
            return LineString(coords)
    return line

# Function to process OSM building elements and extract geometries
def process_building(element):
    outer_ways, inner_ways = [], []  # Initialize as two empty lists
    
    # Process relations
    if isinstance(element, overpy.Relation):
        for member in element.members:
            if isinstance(member, (overpy.Way, overpy.RelationWay)): 
                if member.role == 'outer':
                    outer_ways.append(member)
                elif member.role == 'inner':
                    inner_ways.append(member)

        if not outer_ways:
            return None  # Skip if no outer ways

        outer_polygons, outer_lines = [], []
        for outer_way in outer_ways:
            if isinstance(outer_way, overpy.RelationWay):
                outer_way = outer_way.resolve()

            nodes = [(float(node.lon), float(node.lat)) for node in outer_way.nodes]
            if len(nodes) >= 2:
                line = LineString(nodes)
                outer_lines.append(line)

        try:
            merged_line = linemerge(outer_lines)
            if isinstance(merged_line, LineString):
                merged_line = close_small_gaps(merged_line)
                if merged_line.is_ring:
                    outer_polygon = Polygon(merged_line)
                    if outer_polygon.is_valid:
                        outer_polygons.append(outer_polygon)
                    else:
                        outer_polygon = make_valid(outer_polygon)
                        if outer_polygon.is_valid:
                            outer_polygons.append(outer_polygon)
            elif isinstance(merged_line, MultiPolygon):
                outer_polygons = list(merged_line.geoms)
            else:
                outer_polygons = [Polygon(line) for line in outer_lines if line.is_ring and Polygon(line).is_valid]
        except TopologicalError as e:
            outer_polygons = [Polygon(line) for line in outer_lines if line.is_ring and Polygon(line).is_valid]

        if not outer_polygons:
            return None

        if len(outer_polygons) > 1:
            geometry = max(outer_polygons, key=lambda p: p.area)
        else:
            geometry = outer_polygons[0]

        # Add holes (inner polygons)
        inner_polygons = []
        for inner_way in inner_ways:
            if isinstance(inner_way, overpy.RelationWay):
                inner_way = inner_way.resolve()

            nodes = [(float(node.lon), float(node.lat)) for node in inner_way.nodes]
            if len(nodes) >= 4:
                if nodes[0] != nodes[-1]:
                    nodes.append(nodes[0])
                inner_polygon = Polygon(nodes)
                if inner_polygon.is_valid:
                    inner_polygons.append(inner_polygon)

        try:
            geometry = Polygon(geometry.exterior.coords, holes=[inner.exterior.coords for inner in inner_polygons if geometry.contains(inner)])
        except TopologicalError as e:
            return None

        if not geometry.is_valid:
            geometry = geometry.buffer(0)
            if not geometry.is_valid:
                return None

    # Simple polygons (for ways)
    else:
        nodes = [(float(node.lon), float(node.lat)) for node in element.nodes]
        if len(nodes) >= 4:
            if nodes[0] != nodes[-1]:
                nodes.append(nodes[0])
            try:
                geometry = Polygon(nodes)
                if not geometry.is_valid:
                    geometry = geometry.buffer(0)
                    if not geometry.is_valid:
                        return None
            except TopologicalError as e:
                return None
        else:
            return None

    height = element.tags.get('height')
    levels = element.tags.get('building:levels')
    building_type = element.tags.get('building')
    name = element.tags.get('name')

    if height:
        try:
            height = float(height)
        except ValueError:
            height = None

    return {
        'osm_id': element.id,
        'geometry': geometry,
        'height': height,
        'levels': levels,
        'building_type': building_type,
        'name': name
    }

# Function to get facade midpoints for each building's polygon
def get_facade_midpoints_per_building(polygon, building_id):
    midpoints = []
    exterior_coords = list(polygon.exterior.coords)
    
    # Process each edge of the polygon and compute its midpoint
    for i in range(len(exterior_coords) - 1):
        p1 = exterior_coords[i]
        p2 = exterior_coords[i + 1]
        
        # Create midpoint
        midpoint = Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        
        # Store facade midpoint data
        midpoints.append({
            'building_id': building_id, 
            'geometry': midpoint, 
            'facade_index': i
        })
        
    return midpoints

# Query OSM and process building data
lat_min, lon_min, lat_max, lon_max = 44.462038, 11.246704, 44.546981, 11.425919
api = overpy.Overpass()
query = f"""
[out:json][timeout:25];
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

# Create a GeoDataFrame for building footprints
gdf = gpd.GeoDataFrame(buildings, crs="EPSG:4326")

# Filter for valid geometries
gdf = gdf[gdf.geometry.is_valid]

# Reproject to target CRS (EPSG:7791 - RDN2008 / UTM zone 32N)
target_crs = "EPSG:7791"
gdf_projected = gdf.to_crs(target_crs)

# Filter out small buildings (e.g., less than 10 square meters)
min_area = 10  # in square meters
gdf_projected = gdf_projected[gdf_projected.geometry.area > min_area]

# Ensure all geometries are valid
gdf = gdf[gdf.geometry.is_valid]

# Process each building footprint separately
facade_midpoints_data = []
for index, row in gdf_projected.iterrows():  # Iterate through rows with index
    building_id = row['osm_id'] if 'osm_id' in gdf_projected.columns else index  # Use OSM ID if available, otherwise use row index
    polygon = row['geometry']
    midpoints = get_facade_midpoints_per_building(polygon, building_id)
    facade_midpoints_data.extend(midpoints)

# Create a GeoDataFrame for facade midpoints
facade_midpoints_gdf = gpd.GeoDataFrame(facade_midpoints_data, crs=gdf_projected.crs)

# Plot to verify midpoints per building
fig, ax = plt.subplots(figsize=(12, 12))

# Plot all building footprints in grey
gdf_projected.plot(ax=ax, color='lightgrey', edgecolor='grey')

# Plot facade midpoints in red
facade_midpoints_gdf.plot(ax=ax, color='red', markersize=3)

plt.title("Building Footprints and Facade Midpoints for Each Building")
plt.show()

# Parameters
buffer_distance = 0.1  # For joining buildings
min_segment_length = 1.0  # Minimum length for facade segments

# Use spatial join to find touching buildings (faster than iterating through all pairs)
gdf_projected['temp_id'] = gdf_projected.index
gdf_buffered = gdf_projected.copy()
gdf_buffered['geometry'] = gdf_projected.geometry.buffer(buffer_distance)
joined_gdf = gpd.sjoin(gdf_buffered, gdf_buffered, how='inner', predicate='intersects')

# Create the graph from the spatial join results
G = nx.Graph()
for _, row in joined_gdf.iterrows():
    G.add_edge(row['temp_id_left'], row['temp_id_right'])

# Find connected components (groups of adjacent buildings)
connected_components = list(nx.connected_components(G))

# Assign group IDs to buildings based on connected components
gdf_projected['group_id'] = None
for group_id, component in enumerate(connected_components):
    if len(component) == 1:
        # This is a standalone building
        gdf_projected.loc[list(component), 'group_id'] = f'standalone_{group_id}'
    else:
        gdf_projected.loc[list(component), 'group_id'] = f'block_{group_id}'

# Group buildings into groups
groups = gdf_projected.groupby('group_id')

# Process each group and identify facade segments along the outer edge, including building IDs
facade_segments = []
exterior_boundaries = []  # To store exterior boundaries for plotting

for group_id, group_gdf in groups:
    # Merge all building polygons in the group using unary_union
    merged_polygon = unary_union(group_gdf['geometry'])

    # Process the merged polygon based on its type
    if merged_polygon.geom_type == 'MultiPolygon':
        polygons = list(merged_polygon.geoms)
    elif merged_polygon.geom_type == 'Polygon':
        polygons = [merged_polygon]
    else:
        continue  # Skip groups with invalid geometries

    for polygon in polygons:
        # Simplify the exterior boundary
        exterior_boundary = polygon.exterior

        # Store the exterior boundary for plotting
        exterior_boundaries.append({'group_id': group_id, 'geometry': exterior_boundary})

        # Iterate through the segments of the exterior boundary
        for i in range(len(exterior_boundary.coords) - 1):
            p1 = exterior_boundary.coords[i]
            p2 = exterior_boundary.coords[i + 1]
            segment = LineString([p1, p2])

            # Check if the segment is long enough and part of a building facade
            if segment.length >= min_segment_length:
                building_id = None
                for building_id, building in group_gdf.iterrows():
                    if building['geometry'].intersects(segment):
                        building_id = building['osm_id']
                        break
                if building_id:
                    midpoint = segment.centroid
                    facade_segments.append({
                        'group_id': group_id, 
                        'building_id': building_id, 
                        'geometry': midpoint
                    })

# Create GeoDataFrames for facade segments and exterior boundaries
facade_segments_gdf = gpd.GeoDataFrame(
    {'geometry': [segment['geometry'] for segment in facade_segments],
     'group_id': [segment['group_id'] for segment in facade_segments],
     'building_id': [segment['building_id'] for segment in facade_segments]},
    crs=gdf_projected.crs
)

exterior_boundaries_gdf = gpd.GeoDataFrame(exterior_boundaries, crs=gdf_projected.crs)

# Plot all buildings, boundaries, and midpoints on one map
fig, ax = plt.subplots(figsize=(12, 12))

# Plot all building footprints in grey
gdf_projected.plot(ax=ax, color='lightgrey', edgecolor='grey')

# Plot exterior boundaries in blue
exterior_boundaries_gdf.plot(ax=ax, color='blue', linewidth=1)

# Plot facade midpoints in red
facade_segments_gdf.plot(ax=ax, color='red', markersize=5)

# Add a basemap (you might need to adjust the zoom level and source)
cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom=15)

plt.title("Building Footprints, Group Boundaries, and Facade Midpoints")
plt.show()