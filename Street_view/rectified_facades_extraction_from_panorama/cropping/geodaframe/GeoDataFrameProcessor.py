from cropping.utils.libraries import *

# Helper Functions for GeoDataFrame Processing
class GeoDataFrameProcessor:
    """Utilities for processing GeoDataFrame with building geometry."""
    
    def __init__(self, s3_client):
        self.s3_client = s3_client
    
    def load_geojson(self, path: str) -> gpd.GeoDataFrame:
        """Read a GeoJSON file and return a GeoDataFrame."""
        try:
            return self.s3_client.read_geodataframe('data', path)
        except Exception as exc:
            raise SystemExit(f"Unable to read GeoJSON → {exc}") from exc
    
    @staticmethod
    def ensure_outer_edges(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Ensure geometry_outer_edge_shapely contains Shapely LineStrings."""
        if "geometry_outer_edge_shapely" in gdf.columns:
            return gdf

        if "geometry_outer_edge" not in gdf.columns:
            raise ValueError("GeoJSON has no geometry_outer_edge column.")

        edges: List[LineString] = []
        for geom in gdf["geometry_outer_edge"]:
            if isinstance(geom, LineString):
                edges.append(geom)
            else:
                edges.append(wkt.loads(geom))
        gdf["geometry_outer_edge_shapely"] = edges
        return gdf
    
    @staticmethod
    def ensure_midpoints(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Create geometry_midpoint column if missing."""
        if "geometry_midpoint" in gdf.columns:
            return gdf

        if {"midpoint_x", "midpoint_y"}.issubset(gdf.columns):
            midpoints = [Point(xy) for xy in zip(gdf["midpoint_x"], gdf["midpoint_y"])]
            gdf["geometry_midpoint"] = midpoints
            return gdf

        gdf = GeoDataFrameProcessor.ensure_outer_edges(gdf)
        gdf["geometry_midpoint"] = gdf["geometry_outer_edge_shapely"].centroid
        return gdf
    
    @staticmethod
    def yaw_radians(cam_x: float, cam_y: float, tgt_x: float, tgt_y: float) -> float:
        """Calculate azimuth (radians) from camera to target using atan2 convention."""
        return math.atan2(tgt_y - cam_y, tgt_x - cam_x)
    
    @staticmethod
    def first_two_vertices(edge: LineString) -> Tuple[Point, Point | None]:
        """Return the first and second vertices as Points (second may be None)."""
        coords = list(edge.coords)
        first_pt = Point(coords[0])
        second_pt = Point(coords[1]) if len(coords) > 1 else None
        return first_pt, second_pt

    @staticmethod
    def calculate_cardinal_direction(source: string, target: string) -> string:
        """Calculate the cardinal direction from source to target."""
        
        map_directions = {
            "N": 0,
            "E": 90,
            "S": 180,
            "W": 270,
        }

        source_angle = map_directions[source]
        target_angle = map_directions[target]

        diff = (target_angle - source_angle) % 360

        if diff == 0:
            return "forward"
        elif diff == 180:
            return "backward"
        elif 0 < diff < 180:
            return "right"
        else:  # 180 < diff < 360
            return "left"

