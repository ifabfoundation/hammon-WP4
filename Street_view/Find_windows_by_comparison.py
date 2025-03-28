# %%
import cv2
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import os
import string

# %%
# Load the GeoJSON
gdf_combined = gpd.read_file('../data/Street_view/Preprocessed_to_use_for_picture_extraction.geojson')

# Recreate geometry_midpoint from midpoint_x and midpoint_y
gdf_combined['geometry_midpoint'] = gdf_combined.apply(lambda row: Point(row['midpoint_x'], row['midpoint_y']), axis=1)

# Convert the geometry_outer_edge from string to LineString
gdf_combined['geometry_outer_edge'] = gdf_combined['geometry_outer_edge'].apply(wkt.loads)


# %%
gdf_combined

# %%
def calculate_yaw_pitch(camera_coords, target_coords):
    x1, y1, z1 = camera_coords
    x2, y2, z2 = target_coords

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    horizontal_distance = np.sqrt(dx**2 + dy**2)
    
    if horizontal_distance < 1e-6:
        print("Warning: Camera and target are very close or directly above/below each other.")
        return 0, 90 if dz >= 0 else -90

    yaw = np.arctan2(-dy, -dx)
    yaw_deg = np.degrees(yaw)
    yaw_deg = (yaw_deg + 360) % 360
    
    pitch = np.arctan2(dz, horizontal_distance)
    pitch_deg = np.degrees(pitch)
    
    return yaw_deg, pitch_deg


def enhanced_equirectangular_to_perspective(img, fov, theta, phi, roll, height, width, output_scale=2):
    fov, theta, phi, roll = map(np.radians, (fov, theta, phi, roll))
    
    h, w = img.shape[:2]
    
    # Increase output size
    height *= output_scale
    width *= output_scale
    
    f = width / (2 * np.tan(fov / 2))
    
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    y, x = np.indices((height, width))
    x = (x - width / 2) / f
    y = (y - height / 2) / f
    
    directions = np.dstack([x, y, np.ones_like(x)])
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    
    directions = directions @ R.T
    
    theta_p = np.arctan2(directions[:,:,0], directions[:,:,2])
    phi_p = np.arcsin(np.clip(directions[:,:,1], -1, 1))
    
    u = (theta_p / (2 * np.pi) + 0.5) * w
    v = (phi_p / np.pi + 0.5) * h
    
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)
    
    u_floor, v_floor = np.floor(u).astype(int), np.floor(v).astype(int)
    u_ceil, v_ceil = np.minimum(u_floor + 1, w - 1), np.minimum(v_floor + 1, h - 1)
    
    wu, wv = u - u_floor, v - v_floor
    
    top_left = img[v_floor, u_floor]
    top_right = img[v_floor, u_ceil]
    bottom_left = img[v_ceil, u_floor]
    bottom_right = img[v_ceil, u_ceil]
    
    output = (1 - wu[:,:,np.newaxis]) * ((1 - wv[:,:,np.newaxis]) * top_left + wv[:,:,np.newaxis] * bottom_left) + \
              wu[:,:,np.newaxis] * ((1 - wv[:,:,np.newaxis]) * top_right + wv[:,:,np.newaxis] * bottom_right)
    
    return output.astype(np.uint8)


def calculate_dynamic_fov(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    fov_radians = np.arccos(dot_product / magnitude_product)
    fov_degrees = np.degrees(fov_radians)
    
    fov = min(fov_degrees, 100)
    return fov


def calculate_vectors(camera_coords, target_coords):
    camera = np.array(camera_coords)
    target = np.array(target_coords)
    
    v2 = target - camera
    v1 = np.array([1, 0, 0])
    
    return v1, v2


def adjust_fov_based_on_distance(camera_coords, target_coords, calculated_fov, facade_length):
    x1, y1, _ = camera_coords
    x2, y2, _ = target_coords
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if facade_length < 10:
        if 3.5 < distance < 5:
            adjusted_fov = 90
        elif distance < 3.5:
            adjusted_fov = 95
        else:
            adjusted_fov = calculated_fov
    else:
        if 3.5 < distance < 5:
            adjusted_fov = 100
        elif distance < 3.5:
            adjusted_fov = 105
        else:
            adjusted_fov = calculated_fov
    return adjusted_fov, distance

def project_point_onto_segment(camera, facade_start, facade_end):
    camera = np.array(camera)
    facade_start = np.array(facade_start)
    facade_end = np.array(facade_end)
    
    AB = facade_end - facade_start
    AP = camera - facade_start
    
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return facade_start
    
    t = np.dot(AP, AB) / AB_squared
    t = np.clip(t, 0, 1)
    
    projected_point = facade_start + t * AB
    return projected_point


def evaluate_fov_for_multiple_points(camera_coords, midpoint_coords, facade_start, facade_end, z_value, facade_length):
    """
    Evaluate the FOV for 6 points: the facade extremes, midpoint, projection, and the midpoints between the midpoint and the extremes.
    Return the point with the lowest FOV.
    """
    # Define the six points to evaluate
    points = {
        "midpoint": midpoint_coords,
        "facade_end": np.array([facade_end[0], facade_end[1], z_value]),
        "projected": np.array([*project_point_onto_segment(camera_coords[:2], facade_start, facade_end), z_value]),
        "mid_mid_start": np.array([(midpoint_coords[0] + facade_start[0]) / 2, (midpoint_coords[1] + facade_start[1]) / 2, z_value]),
        "mid_mid_end": np.array([(midpoint_coords[0] + facade_end[0]) / 2, (midpoint_coords[1] + facade_end[1]) / 2, z_value])
    }
    
    # Evaluate FOV for each point
    fov_results = []
    projected_result = None
    for point_name, target_coords in points.items():
        yaw, pitch = calculate_yaw_pitch(camera_coords, target_coords)
        v1, v2 = calculate_vectors(camera_coords, target_coords)
        fov = calculate_dynamic_fov(v1, v2)
        fov_adjusted, distance = adjust_fov_based_on_distance(camera_coords, target_coords, fov, facade_length)
        
        # Store the result for each point
        fov_results.append((fov, fov_adjusted, target_coords, yaw, pitch, point_name))
        
        # Capture the projected point's result
        if point_name == "projected":
            projected_result = (fov_adjusted, target_coords, yaw, pitch, distance)

    # If distance is less than 5 meters, return the projected point with adjusted FOV
    if projected_result[4] < 5:
        print(f"Using projection-based method due to close distance (<5m).")
        print(f"Best FOV: {projected_result[0]} from projected")
        return projected_result[1], projected_result[2], projected_result[3], projected_result[0]
    elif projected_result[4] > 15:
        print("Facade too far from camera, probably part of block not facing street")
        return None, None, None, None
    
    # Otherwise, work with facade length, if lower than 10 m it is ok to minimiz fov, otherwise maximize it
    if facade_length < 10:
        print("Maximizing FOV (facade length <10m).")
        best_fov, best_fov_adjusted, best_target, best_yaw, best_pitch, best_point_name = min(fov_results, key=lambda x: x[1])
    else:
        print("Minimizing FOV (facade length >10m).")
        best_fov, best_fov_adjusted, best_target, best_yaw, best_pitch, best_point_name = max(fov_results, key=lambda x: x[1])
    # if fov still less than 80, use 90
    if best_fov_adjusted < 80:
        print("Setting FOV to 90.")
        best_fov_adjusted = 90
    print(f"Best FOV: {best_fov_adjusted} from {best_point_name}")
    
    return best_target, best_yaw, best_pitch, best_fov_adjusted


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_pitch_adjustment_based_on_distance(camera_coords, midpoint_coords, base_pitch):
    x1, y1, _ = camera_coords
    x2, y2, _ = midpoint_coords
    
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if distance < 3.5:
        pitch_adjustment = base_pitch + 25
    elif distance < 5:
        pitch_adjustment = base_pitch + 25
    else:
        pitch_adjustment = base_pitch + 25 
    
    return pitch_adjustment

# %%
# Main loop to apply the 6-fold FOV evaluation and pitch adjustment
plt.close('all')
%matplotlib inline

images_folder = '../data/PANO_new/'
output_folder = '../data/Street_view/Output_images/'
# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Dictionary to track how many images we've saved per building
building_image_count = {}

for i, row in gdf_combined.iterrows():
    camera_coords = (row.loc['X'], row.loc['Y'], row.loc['Z'])
    midpoint_coords = (row.loc['geometry_midpoint'].x, row.loc['geometry_midpoint'].y, row.loc['Z'] + 0.25) 

    # Calculate facade extremes and midpoint
    facade_start = (row.loc['geometry_outer_edge'].coords[0][:2])
    facade_end = (row.loc['geometry_outer_edge'].coords[-1][:2])

    facade_length = np.linalg.norm(np.array([facade_end[0], facade_end[1]]) - np.array([facade_start[0], facade_start[1]]))
    print(f"Facade length: {facade_length}")
    
    target_z = midpoint_coords[2]
    
    # Evaluate the FOV for multiple points and select the best
    target_coords, yaw, pitch, fov = evaluate_fov_for_multiple_points(camera_coords, midpoint_coords, facade_start, facade_end, target_z, facade_length)
    if target_coords is None:
        continue

    # Apply dynamic pitch adjustment based on the distance
    pitch = calculate_pitch_adjustment_based_on_distance(camera_coords, midpoint_coords, pitch)

    # Load the image
    path = row.loc['PATH']
    run = row.loc['RUN']
    image_name = row.loc['FOTO']
    image_path = '{}{}/{}/{}'.format(images_folder, path, run, image_name)
    image = cv2.imread(image_path)
    
    rgb_image = convert_bgr_to_rgb(image)
    roll = row.loc['Roll']

    # Define output image dimensions
    output_width = 1024
    output_height = 1024

    # Apply the perspective transformation
    output_image = enhanced_equirectangular_to_perspective(rgb_image, fov, yaw, pitch, roll, output_height, output_width)

    # Handle filename logic
    building_id = str(row.loc['building_id'])
    if building_id not in building_image_count:
        building_image_count[building_id] = 0
    
    building_image_count[building_id] += 1
    suffix = ''
    
    # If more than one image for the building, append a letter (A, B, C, etc.)
    if building_image_count[building_id] > 1:
        suffix = string.ascii_uppercase[building_image_count[building_id] - 2]

    # Create the filename with the building ID and the appropriate suffix
    output_filename = f"Building_{building_id}{suffix}.png"
    output_path = os.path.join(output_folder, output_filename)

    # Save the image
    plt.imsave(output_path, output_image)

    # Add the output filename to the GeoDataFrame
    gdf_combined.at[i, 'output_filename'] = output_filename

    # Display results
    print(row.loc['group_id'], row.loc['building_id'], row.loc['osm_id'], row.loc['geometry_midpoint'], row.loc['orientation_midpoint'], image_name)
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

    plt.close('all')

# Drop unwanted geometry columns before saving
gdf_combined = gdf_combined.drop(columns=['geometry_midpoint'])  # Drop 'geometry_midpoint' if keeping 'geometry'

# Set 'geometry' as the active geometry column
gdf_combined = gdf_combined.set_geometry('geometry')

# Save the GeoDataFrame to a file
gdf_combined.to_file('data/Street_view/Preprocessed_with_filenames.geojson', driver='GeoJSON')

# %%



