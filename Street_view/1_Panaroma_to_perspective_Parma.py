# %%
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely import wkt
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
import string # For filename suffixes

# --- Configuration Parameters ---
INPUT_GEOJSON = '../data/Street_view/Preprocessed_to_use_for_picture_extraction.geojson'
OUTPUT_GEOJSON = 'data/Street_view/Preprocessed_with_filenames_REVISED.geojson'
IMAGES_BASE_FOLDER = '../data/PANO_new/'
OUTPUT_IMAGES_FOLDER = '../data/Street_view/Output_images_REVISED/'

# Camera/Targeting Settings
CAMERA_HEIGHT_OFFSET = 0.0 # Assume Z in GeoJSON is camera height. Adjust if Z is ground.
TARGET_Z_OFFSET = 0.25 # How much higher than camera Z to aim at the target (e.g., slightly above midpoint)
PITCH_OFFSET_DEG = 0.0 # Additional pitch angle (DEGREES). KEEP SMALL (0-5). Avoid large values like 25!

# FoV Calculation Settings
FOV_PADDING_FACTOR = 1.15 # Multiplier for calculated FoV (1.1 = 10% padding)
MIN_FOV_DEG = 50.0 # Minimum allowed horizontal FoV
MAX_FOV_DEG = 100.0 # Maximum allowed horizontal FoV

# Filtering Settings
MIN_DISTANCE_M = 3.0 # Minimum distance from camera to facade target point
MAX_DISTANCE_M = 25.0 # Maximum distance (facades further than this are likely low-res)
FILTER_BY_VIEW_ANGLE = True # Enable filtering based on viewing angle?
MAX_VIEW_ANGLE_DEG = 60.0 # Maximum angle (degrees) between view vector and facade normal (0=fronto-parallel)

# Output Image Settings
OUTPUT_WIDTH = 1024 # Desired width of the output perspective image
OUTPUT_HEIGHT = 1024 # Desired height
OUTPUT_UPSCALE_FACTOR = 1 # Factor to increase internal rendering resolution (1=no upscale, 2=render at 2x then downscale - slower but smoother)

# --- Helper Functions ---

def calculate_yaw_pitch(camera_pos_3d, target_pos_3d):
    """Calculates Yaw (Azimuth) and Pitch (Elevation) from camera to target.

    Args:
        camera_pos_3d (tuple): (x, y, z) of the camera.
        target_pos_3d (tuple): (x, y, z) of the target point.

    Returns:
        tuple: (yaw_deg, pitch_deg)
               Yaw: Angle in degrees [0, 360], relative to the coordinate system's
                    positive X-axis (adjust if needed based on panorama format).
               Pitch: Angle in degrees [-90, 90], relative to the XY plane.
    """
    x1, y1, z1 = camera_pos_3d
    x2, y2, z2 = target_pos_3d

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    horizontal_distance = math.sqrt(dx**2 + dy**2)

    if horizontal_distance < 1e-6: # Avoid division by zero if target is directly above/below
        print("Warning: Camera and target X,Y are virtually identical.")
        yaw_deg = 0 # Yaw is undefined, default to 0
        pitch_deg = 90 if dz >= 0 else -90
    else:
        # Yaw: Angle in the XY plane. Using atan2(dy, dx) gives angle relative to +X axis.
        # Check if your panorama theta=0 corresponds to +X (East) or +Y (North).
        # The original used atan2(-dy, -dx) which is angle relative to -X axis + 180 deg.
        # Let's use atan2(dy, dx) assuming theta=0 is East (+X). Adjust if needed.
        yaw_rad = math.atan2(dy, dx)
        yaw_deg = math.degrees(yaw_rad)
        yaw_deg = (yaw_deg + 360) % 360 # Ensure range [0, 360]

        # Pitch: Angle relative to the horizontal plane.
        pitch_rad = math.atan2(dz, horizontal_distance)
        pitch_deg = math.degrees(pitch_rad)

    return yaw_deg, pitch_deg

def enhanced_equirectangular_to_perspective(img, fov_deg, yaw_deg, pitch_deg, roll_deg,
                                            out_height, out_width, output_scale=1):
    """Warps an equirectangular image to a perspective view.

    Args:
        img: Input equirectangular image (numpy array, HxWxC).
        fov_deg: Desired horizontal field of view (degrees).
        yaw_deg: Center view direction yaw (azimuth, degrees).
        pitch_deg: Center view direction pitch (elevation, degrees).
        roll_deg: View rotation around the viewing axis (degrees).
        out_height: Desired output height in pixels.
        out_width: Desired output width in pixels.
        output_scale: Factor to upscale rendering resolution for anti-aliasing.

    Returns:
        Numpy array: The perspective image.
    """
    fov_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, (fov_deg, yaw_deg, pitch_deg, roll_deg))

    in_h, in_w = img.shape[:2]

    # Render at higher resolution if output_scale > 1
    render_width = out_width * output_scale
    render_height = out_height * output_scale

    # Calculate focal length from FoV and render width
    focal_length = render_width / (2 * np.tan(fov_rad / 2))

    # --- 3D Rotation Setup ---
    # Rotation order: Roll -> Pitch -> Yaw (applied to camera frame)
    # Or equivalently: -Yaw -> -Pitch -> -Roll (applied to world points)
    # We map output pixels (camera frame) to world directions, then to panorama coords.

    # Rotation matrices (using scipy/transforms3d might be cleaner, but numpy is fine)
    # Note: Positive pitch looks up, positive yaw turns left (if theta=0 is North) or right (if theta=0 is East). Check convention.
    # Assuming yaw increases counter-clockwise from +X axis (East)
    # Assuming pitch increases upwards from XY plane
    # Assuming roll increases counter-clockwise around viewing axis

    # Rz(roll) rotates around viewing axis (initially Z)
    # Rx(pitch) rotates around camera's X axis
    # Ry(yaw) rotates around camera's Y axis
    # Let's use ZYX convention for Tait-Bryan angles (Yaw around Z, Pitch around Y, Roll around X)
    # Adjust based on exact definition of yaw, pitch, roll in your system/panorama.
    # If yaw is azimuth (around Z), pitch is elevation (around Y'), roll is around X''

    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)

    # Rotation matrix: camera coordinates -> world coordinates
    # Assuming initial camera looks along +X, Y is left, Z is up. Rotate to target.
    # R = R_z(yaw) @ R_y(pitch) @ R_x(roll)  -- check this order matches your system
    # If yaw is around vertical (Z), pitch around horizontal (Y'), roll around view (X'')
    R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) # Rotation around Z
    R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) # Rotation around Y
    R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]) # Rotation around X

    # Combined rotation: first roll, then pitch, then yaw
    R = R_yaw @ R_pitch @ R_roll
    # If initial camera looks along +Z, Y is right, X is down (OpenGL style): adjust R

    # --- Map Output Pixels to Panorama ---
    # Create grid of pixel coordinates in the output image (rendering resolution)
    y_render, x_render = np.indices((render_height, render_width), dtype=np.float32)

    # Center and scale pixels to represent normalized coordinates on the image plane
    x_cam = (x_render - render_width / 2) / focal_length
    y_cam = (y_render - render_height / 2) / focal_length # Y points downwards in image coords usually
    z_cam = np.ones_like(x_cam) # Assume image plane at z=1

    # Create 3D direction vectors in the camera's coordinate system
    # If camera looks along +X: directions = [z_cam, x_cam, -y_cam] ? Check axis definitions
    # Assuming standard computer vision camera: +X right, +Y down, +Z forward
    directions_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

    # Normalize direction vectors
    directions_cam /= np.linalg.norm(directions_cam, axis=2, keepdims=True)

    # Rotate direction vectors from camera frame to world frame using transpose (inverse) of R
    # directions_world = directions_cam @ R.T # Check if R maps world->cam or cam->world
    # Let's assume R maps *world axes* to *camera axes*. To rotate *camera vector* to *world vector*, use R.T
    # If R maps *camera axes* to *world axes*, use R directly. Let's try R.T
    directions_world = np.einsum('ijk,lk->ijl', directions_cam, R) # Efficient matrix multiplication for batches

    # --- Convert World Directions to Panorama Spherical Coordinates ---
    # Assuming panorama coords: theta=azimuth (0 at +X?), phi=elevation (0 at horizon?)
    x_world = directions_world[:, :, 0]
    y_world = directions_world[:, :, 1]
    z_world = directions_world[:, :, 2]

    # Theta (Azimuth): Use atan2(y, x) for angle from +X axis. Adjust if needed.
    theta_pano = np.arctan2(y_world, x_world) # Radians in [-pi, pi]

    # Phi (Elevation): Use asin(z / radius). Radius is 1 since directions are normalized.
    phi_pano = np.arcsin(np.clip(z_world, -1.0, 1.0)) # Radians in [-pi/2, pi/2]

    # --- Map Spherical Coordinates to Equirectangular Pixel Coordinates ---
    # u: horizontal pixel coord (0 to in_w). theta=0 -> u=in_w/2 ? theta=-pi -> u=0?
    # Assuming theta=0 at center (+X), theta ranges [-pi, pi] maps to [0, in_w]
    u = (theta_pano / (2 * np.pi) + 0.5) * in_w # Maps [-pi, pi] to [0, in_w]

    # v: vertical pixel coord (0 to in_h). phi=pi/2 -> v=0? phi=-pi/2 -> v=in_h?
    # Assuming phi=pi/2 (North Pole) at v=0, phi=-pi/2 (South Pole) at v=in_h
    v = (-phi_pano / np.pi + 0.5) * in_h # Maps [pi/2, -pi/2] to [0, in_h]

    # --- Sample Pixels using Bilinear Interpolation ---
    # (Using cv2.remap is often faster and handles boundaries better)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    # cv2.remap expects map_x, map_y with dimensions (out_H, out_W)
    # Interpolation method
    interpolator = cv2.INTER_LINEAR

    # Use cv2.remap for potentially faster interpolation
    output_render = cv2.remap(img, map_x, map_y,
                              interpolation=interpolator,
                              borderMode=cv2.BORDER_WRAP) # Use WRAP for 360 images

    # Downscale if rendered at higher resolution
    if output_scale > 1:
        output_image = cv2.resize(output_render, (out_width, out_height), interpolation=cv2.INTER_AREA)
    else:
        output_image = output_render

    return output_image.astype(np.uint8) # Ensure output is uint8

def project_point_onto_segment(point_xy, seg_start_xy, seg_end_xy):
    """Projects a 2D point onto a 2D line segment."""
    point = np.array(point_xy)
    start = np.array(seg_start_xy)
    end = np.array(seg_end_xy)

    line_vec = end - start
    point_vec = point - start

    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 1e-9: # Segment is a point
        return start

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1) # Clamp projection to the segment

    projected_point = start + t * line_vec
    return projected_point

def calculate_facade_normal(facade_segment):
    """Estimates the 2D normal vector of a facade LineString (pointing outwards)."""
    coords = list(facade_segment.coords)
    if len(coords) < 2:
        return None # Cannot calculate normal for a point

    # Use the first segment for direction (or average if needed)
    p1 = np.array(coords[0][:2])
    p2 = np.array(coords[1][:2])

    # Direction vector along the facade
    dir_vec = p2 - p1
    dir_vec_norm = np.linalg.norm(dir_vec)
    if dir_vec_norm < 1e-6:
        return None # Segment too short

    # Normalized direction vector
    dir_vec = dir_vec / dir_vec_norm

    # Normal vector (rotate 90 degrees) - potential ambiguity in direction (in/out)
    # Option 1: Rotate counter-clockwise
    normal_vec = np.array([-dir_vec[1], dir_vec[0]])
    # Option 2: Rotate clockwise
    # normal_vec = np.array([dir_vec[1], -dir_vec[0]])

    # We need a heuristic to determine outwards normal, maybe based on building polygon centroid?
    # For now, just return one possibility. Assuming OSM polygon winding order might help.
    return normal_vec # Needs check for outwards direction if critical

def calculate_geometric_fov(facade_length, distance, padding_factor=1.1, min_fov=60.0, max_fov=100.0):
    """Calculates horizontal FoV needed to capture facade based on distance and length."""
    if distance < 1e-3:
        return max_fov # Very close, use max FoV

    # Angle subtended by half the facade length
    angle_half_facade = math.atan((facade_length / 2.0) / distance)

    # Full angle needed for the facade
    fov_rad = 2 * angle_half_facade

    # Apply padding
    fov_rad *= padding_factor

    fov_deg = math.degrees(fov_rad)

    # Clamp to min/max limits
    fov_deg_clamped = np.clip(fov_deg, min_fov, max_fov)

    return fov_deg_clamped

def convert_bgr_to_rgb(image):
    """Converts BGR image (from cv2.imread) to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Main Processing Logic ---

print(f"Loading GeoJSON: {INPUT_GEOJSON}")
try:
    gdf_combined = gpd.read_file(INPUT_GEOJSON)
except Exception as e:
    print(f"Error loading GeoJSON: {e}")
    exit()

# Prepare geometry columns (handle potential errors)
try:
    if 'midpoint_x' in gdf_combined.columns and 'midpoint_y' in gdf_combined.columns:
        gdf_combined['geometry_midpoint'] = gdf_combined.apply(
            lambda row: Point(row['midpoint_x'], row['midpoint_y']), axis=1
        )
    else:
        print("Warning: 'midpoint_x' or 'midpoint_y' columns not found.")
        # Attempt to calculate from 'geometry_outer_edge' if midpoint missing
        if 'geometry_outer_edge' in gdf_combined.columns:
             gdf_combined['geometry_outer_edge_shapely'] = gdf_combined['geometry_outer_edge'].apply(wkt.loads)
             gdf_combined['geometry_midpoint'] = gdf_combined['geometry_outer_edge_shapely'].centroid
        else:
             raise ValueError("Cannot determine midpoint.")

    if 'geometry_outer_edge' in gdf_combined.columns and not isinstance(gdf_combined['geometry_outer_edge'].iloc[0], LineString):
         gdf_combined['geometry_outer_edge_shapely'] = gdf_combined['geometry_outer_edge'].apply(wkt.loads)
    elif 'geometry_outer_edge_shapely' not in gdf_combined.columns:
         raise ValueError("'geometry_outer_edge' column missing or not WKT.")
    else:
         # If already shapely objects, ensure correct column name
         if 'geometry_outer_edge_shapely' not in gdf_combined.columns:
             gdf_combined.rename(columns={'geometry_outer_edge': 'geometry_outer_edge_shapely'}, inplace=True)


except Exception as e:
    print(f"Error preparing geometry columns: {e}")
    exit()


print(f"Outputting images to: {OUTPUT_IMAGES_FOLDER}")
os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)

# Dictionary to track image counts per building for naming
building_image_count = {}
output_filenames_list = []

print("Starting perspective image extraction...")
for i, row in gdf_combined.iterrows():
    try:
        print(f"\nProcessing row {i} (Building ID: {row.get('building_id', 'N/A')}, OSM ID: {row.get('osm_id', 'N/A')})")

        # --- 1. Get Inputs ---
        cam_x, cam_y, cam_z = row['X'], row['Y'], row['Z'] + CAMERA_HEIGHT_OFFSET
        roll_deg = row['Roll']
        facade_line = row['geometry_outer_edge_shapely']
        facade_midpoint_geom = row['geometry_midpoint']

        if not isinstance(facade_line, LineString) or len(facade_line.coords) < 2:
             print("Skipping: Invalid facade geometry.")
             output_filenames_list.append(None)
             continue

        facade_start_xy = facade_line.coords[0][:2]
        facade_end_xy = facade_line.coords[-1][:2]
        facade_length = facade_line.length

        # --- 2. Determine Target Point ---
        cam_xy = (cam_x, cam_y)
        projected_target_xy = project_point_onto_segment(cam_xy, facade_start_xy, facade_end_xy)

        # Choose target point (e.g., projected point)
        target_x, target_y = projected_target_xy
        target_z = cam_z + TARGET_Z_OFFSET # Aim slightly higher than camera

        camera_pos_3d = (cam_x, cam_y, cam_z)
        target_pos_3d = (target_x, target_y, target_z)

        # --- 3. Calculate Distance and Basic Filtering ---
        dist_xy = math.sqrt((target_x - cam_x)**2 + (target_y - cam_y)**2)
        print(f"  Distance (XY): {dist_xy:.2f}m")

        if not (MIN_DISTANCE_M <= dist_xy <= MAX_DISTANCE_M):
            print(f"Skipping: Distance {dist_xy:.2f}m out of range [{MIN_DISTANCE_M}, {MAX_DISTANCE_M}].")
            output_filenames_list.append(None)
            continue

        # --- 4. View Angle Filtering (Optional) ---
        if FILTER_BY_VIEW_ANGLE:
            facade_normal_vec = calculate_facade_normal(facade_line)
            if facade_normal_vec is not None:
                view_vec_xy = np.array([target_x - cam_x, target_y - cam_y])
                view_vec_norm = np.linalg.norm(view_vec_xy)

                if view_vec_norm > 1e-6:
                    view_vec_xy /= view_vec_norm # Normalize

                    # Dot product between normalized view vector and facade normal
                    # Use absolute value as normal direction might be ambiguous
                    cos_angle = abs(np.dot(view_vec_xy, facade_normal_vec))
                    angle_deg = math.degrees(np.arccos(np.clip(cos_angle, 0, 1))) # Angle is between view and normal line (90=parallel)
                    view_angle_to_normal = 90.0 - angle_deg # Angle between view and facade plane (0=parallel, 90=perpendicular)

                    # We want angle between view vector and *normal* to be small
                    angle_check = angle_deg # This is angle between view vector and the normal vector

                    print(f"  View Angle to Facade Normal: {angle_check:.2f} deg")
                    if angle_check > MAX_VIEW_ANGLE_DEG:
                        print(f"Skipping: View angle {angle_check:.2f} deg > {MAX_VIEW_ANGLE_DEG} deg (too oblique).")
                        output_filenames_list.append(None)
                        continue
            else:
                print("  Warning: Could not calculate facade normal for angle check.")


        # --- 5. Calculate View Parameters (Yaw, Pitch, FoV) ---
        yaw_deg, pitch_deg_raw = calculate_yaw_pitch(camera_pos_3d, target_pos_3d)
        pitch_deg = pitch_deg_raw + PITCH_OFFSET_DEG # Apply small offset if desired
        print(f"  Calculated Yaw: {yaw_deg:.2f}, Raw Pitch: {pitch_deg_raw:.2f}, Final Pitch: {pitch_deg:.2f}")

        fov_deg = calculate_geometric_fov(facade_length, dist_xy, FOV_PADDING_FACTOR, MIN_FOV_DEG, MAX_FOV_DEG)
        print(f"  Facade Length: {facade_length:.2f}m, Calculated FoV: {fov_deg:.2f} deg")

        # --- 6. Load Panorama Image ---
        path_val = row['PATH']
        run_val = row['RUN']
        foto_val = row['FOTO']
        image_path = os.path.join(IMAGES_BASE_FOLDER, str(path_val), str(run_val), str(foto_val))

        if not os.path.exists(image_path):
            print(f"Skipping: Image not found at {image_path}")
            output_filenames_list.append(None)
            continue

        panorama_image_bgr = cv2.imread(image_path)
        if panorama_image_bgr is None:
             print(f"Skipping: Failed to load image {image_path}")
             output_filenames_list.append(None)
             continue
        panorama_image_rgb = convert_bgr_to_rgb(panorama_image_bgr)
        print(f"  Loaded image: {image_path}")

        # --- 7. Extract Perspective Image ---
        output_image = enhanced_equirectangular_to_perspective(
            panorama_image_rgb,
            fov_deg, yaw_deg, pitch_deg, roll_deg,
            OUTPUT_HEIGHT, OUTPUT_WIDTH, OUTPUT_UPSCALE_FACTOR
        )

        # --- 8. Save Output Image ---
        building_id = str(row.get('building_id', f'Idx_{i}')) # Use index if ID missing
        if building_id not in building_image_count:
            building_image_count[building_id] = 0

        building_image_count[building_id] += 1
        suffix = ''
        # Append letter only if it's the second, third, etc., image for this building ID
        if building_image_count[building_id] > 1:
            # Map count 2->A, 3->B, etc.
            suffix = string.ascii_uppercase[building_image_count[building_id] - 2]

        output_filename = f"Building_{building_id}{suffix}.png"
        output_path = os.path.join(OUTPUT_IMAGES_FOLDER, output_filename)

        print(f"  Saving perspective image to: {output_path}")
        saved = plt.imsave(output_path, output_image)
        # Alternative save using OpenCV (might be faster, check color order)
        # saved = cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        if not saved: # Check if saving failed (though plt.imsave doesn't return bool)
             print(f"  Warning: Failed to save image {output_path}")
             output_filenames_list.append(None)
        else:
             output_filenames_list.append(output_filename)

        # Optional: Display results for debugging
        # plt.imshow(output_image)
        # plt.title(f"Bldg {building_id}{suffix} - Dist:{dist_xy:.1f}m, Ang:{angle_check:.1f}, FoV:{fov_deg:.1f}")
        # plt.axis('off')
        # plt.show()
        # plt.close('all') # Close plot window

    except Exception as e:
        print(f"!!!!!!!! ERROR processing row {i}: {e} !!!!!!!!!!")
        import traceback
        traceback.print_exc() # Print detailed traceback
        output_filenames_list.append(None)
        continue # Continue to the next row

# --- Final Step: Update GeoDataFrame and Save ---
print("\nUpdating GeoDataFrame with output filenames...")
# Ensure the list length matches the DataFrame length
if len(output_filenames_list) != len(gdf_combined):
     print(f"Error: Length mismatch between results ({len(output_filenames_list)}) and GeoDataFrame ({len(gdf_combined)}). Saving skipped.")
else:
     gdf_combined['output_filename_revised'] = output_filenames_list

     # Drop temporary geometry columns if they exist
     if 'geometry_midpoint' in gdf_combined.columns:
         gdf_combined = gdf_combined.drop(columns=['geometry_midpoint'])
     if 'geometry_outer_edge_shapely' in gdf_combined.columns:
         # Decide if you want to keep the shapely version or the original WKT
         # Let's drop the temporary shapely one if the original WKT exists
         if 'geometry_outer_edge' in gdf_combined.columns:
              gdf_combined = gdf_combined.drop(columns=['geometry_outer_edge_shapely'])
         else: # Otherwise, maybe rename it if needed?
              pass # gdf_combined.rename(columns={'geometry_outer_edge_shapely': 'geometry'}, inplace=True)


     # Make sure a valid geometry column is set if needed for saving
     if 'geometry' not in gdf_combined.columns and 'geometry_outer_edge_shapely' in gdf_combined.columns:
          gdf_combined = gdf_combined.set_geometry('geometry_outer_edge_shapely')
     elif 'geometry' not in gdf_combined.columns and 'geometry_midpoint' in gdf_combined.columns:
          gdf_combined = gdf_combined.set_geometry('geometry_midpoint')
     # Add more logic if necessary to ensure 'geometry' column exists/is set

     print(f"Saving updated GeoDataFrame to: {OUTPUT_GEOJSON}")
     try:
         # Ensure geometry is set before saving
         if 'geometry' not in gdf_combined.columns:
              print("Warning: No active geometry column set in GeoDataFrame. Saving might fail or lack geometry.")
              # Attempt to set one if possible, e.g., from original input if it exists
              # gdf_combined = gdf_combined.set_geometry('original_geometry_column_name')

         gdf_combined.to_file(OUTPUT_GEOJSON, driver='GeoJSON')
         print("GeoDataFrame saved successfully.")
     except Exception as e:
         print(f"Error saving updated GeoJSON: {e}")

print("\nProcessing finished.")


