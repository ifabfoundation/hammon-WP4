import cv2
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import matplotlib
# Set non-interactive backend to avoid Qt errors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely import wkt
import os
import string
from skimage import feature, color, transform, io
import logging
from skimage import img_as_ubyte, img_as_float

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the GeoJSON
gdf_combined = gpd.read_file('data/Street_view/Preprocessed_to_use_for_picture_extraction.geojson')

# Recreate geometry_midpoint from midpoint_x and midpoint_y
gdf_combined['geometry_midpoint'] = gdf_combined.apply(lambda row: Point(row['midpoint_x'], row['midpoint_y']), axis=1)

# Convert the geometry_outer_edge from string to LineString
gdf_combined['geometry_outer_edge'] = gdf_combined['geometry_outer_edge'].apply(wkt.loads)

# Core functions from Find_windows_by_comparison.py

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


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def calculate_pitch_adjustment_based_on_distance(camera_coords, midpoint_coords, base_pitch):
    x1, y1, _ = camera_coords
    x2, y2, _ = midpoint_coords
    
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    if distance < 5:
        pitch_adjustment = base_pitch + 25
    else:
        pitch_adjustment = base_pitch + 25 
    
    return pitch_adjustment

# Functions for vanishing point analysis

def compute_edgelets(image, sigma=3):
    """Create edgelets as in the paper.

    Uses canny edge detection and then finds (small) lines using probabilistic
    hough transform as edgelets.

    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.

    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """
    # Ensure the image has 3 channels
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[-1] == 4:
        # Remove alpha channel
        image = image[..., :3]

    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=3,
                                               line_gap=2)

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # Convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    if len(directions) > 0:  # Check if any lines were detected
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def filter_edgelets_by_orientation(edgelets, orientation='horizontal', threshold=15):
    """Filter edgelets based on their orientation.
    
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    orientation: str
        'horizontal' or 'vertical'
    threshold: float
        Threshold in degrees from the true horizontal/vertical.
        
    Returns
    -------
    filtered_edgelets: tuple of ndarrays
        Filtered edgelets.
    """
    locations, directions, strengths = edgelets
    
    if len(directions) == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    # Calculate angle with horizontal axis (in degrees)
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angles_deg = np.degrees(angles) % 180
    
    if orientation == 'horizontal':
        # Keep edgelets close to 0 or 180 degrees (horizontal)
        mask = (angles_deg < threshold) | (angles_deg > (180 - threshold))
    else:  # vertical
        # Keep edgelets close to 90 degrees (vertical)
        mask = (angles_deg > (90 - threshold)) & (angles_deg < (90 + threshold))
    
    return (locations[mask], directions[mask], strengths[mask])


def evaluate_horizontal_parallelism(edgelets):
    """Evaluate how parallel the horizontal edgelets are.
    
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        
    Returns
    -------
    parallelism_score: float
        Score indicating how parallel the edgelets are (higher is better).
    """
    _, directions, strengths = edgelets
    
    if len(directions) < 2:
        return 0  # Not enough lines to evaluate parallelism
    
    # Calculate angles with horizontal axis
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angles_deg = np.degrees(angles) % 180
    
    # Calculate standard deviation of angles (weighted by strengths)
    # Lower std dev means more parallel lines
    weighted_std = np.sqrt(np.average((angles_deg - np.average(angles_deg, weights=strengths))**2, weights=strengths))
    
    # Convert to a score (higher is better)
    parallelism_score = 1 / (1 + weighted_std)
    
    return parallelism_score


def evaluate_horizontal_alignment(edgelets):
    """Evaluate how well-aligned the horizontal edgelets are with true horizontal.
    
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        
    Returns
    -------
    alignment_score: float
        Score indicating how well-aligned the edgelets are (higher is better).
    """
    _, directions, strengths = edgelets
    
    if len(directions) == 0:
        return 0  # No lines to evaluate
    
    # Calculate angles with horizontal axis
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angles_deg = np.degrees(angles) % 180
    
    # Calculate how close angles are to 0 or 180 degrees (true horizontal)
    deviations = np.minimum(angles_deg, 180 - angles_deg)
    
    # Calculate weighted average deviation (weighted by strengths)
    weighted_avg_deviation = np.average(deviations, weights=strengths)
    
    # Convert to a score (higher is better)
    alignment_score = 1 / (1 + weighted_avg_deviation)
    
    return alignment_score


def evaluate_perspective_image(image):
    """Evaluate a perspective image for rectification suitability.
    
    Parameters
    ----------
    image: ndarray
        Perspective image to evaluate.
        
    Returns
    -------
    score_dict: dict
        Dictionary containing various scores and metrics.
    """
    # Compute all edgelets
    edgelets = compute_edgelets(image, sigma=1.5)
    
    # Filter horizontal edgelets
    horizontal_edgelets = filter_edgelets_by_orientation(edgelets, orientation='horizontal', threshold=20)
    
    # Calculate scores
    parallelism_score = evaluate_horizontal_parallelism(horizontal_edgelets)
    alignment_score = evaluate_horizontal_alignment(horizontal_edgelets)
    
    # Count strong edgelets
    _, _, strengths = horizontal_edgelets
    strong_edgelets_count = np.sum(strengths > 5) if len(strengths) > 0 else 0
    
    # Density of horizontal lines (normalized by image height)
    h, w = image.shape[:2]
    horizontal_line_density = len(strengths) / h if len(strengths) > 0 else 0
    
    # Combined score (can be adjusted based on what's most important)
    combined_score = (
        0.4 * parallelism_score + 
        0.4 * alignment_score + 
        0.1 * (strong_edgelets_count / 100) +  # Normalize to roughly 0-1 range
        0.1 * horizontal_line_density
    )
    
    return {
        'parallelism_score': parallelism_score,
        'alignment_score': alignment_score,
        'strong_edgelets_count': strong_edgelets_count,
        'horizontal_line_density': horizontal_line_density,
        'combined_score': combined_score,
        'horizontal_edgelets_count': len(horizontal_edgelets[0])
    }


def evaluate_perspective_images_for_points(image, camera_coords, points, z_value, roll=0, output_width=1024, output_height=1024):
    """
    Generate and evaluate perspective images for multiple target points.
    Return the best image based on vanishing point quality.
    
    Parameters
    ----------
    image: ndarray
        Panorama image.
    camera_coords: tuple
        (x, y, z) coordinates of the camera.
    points: dict
        Dictionary of point names to their (x, y, z) coordinates.
    z_value: float
        Z-value for points that don't have it.
    roll: float
        Roll angle for the camera.
        
    Returns
    -------
    best_result: dict
        Information about the best perspective image.
    """
    results = []
    
    for point_name, target_coords in points.items():
        # Ensure target has z-value
        if len(target_coords) == 2:
            target_coords = (*target_coords, z_value)
            
        # Calculate camera parameters
        yaw, pitch = calculate_yaw_pitch(camera_coords, target_coords)
        v1, v2 = calculate_vectors(camera_coords, target_coords)
        fov = calculate_dynamic_fov(v1, v2)
        
        # Adjust FOV based on heuristics
        # Start with moderate FOV that should capture good building details
        fov_adjusted = min(max(fov, 80), 110)
        
        # Generate perspective image
        perspective_img = enhanced_equirectangular_to_perspective(
            image, fov_adjusted, yaw, pitch, roll, output_height, output_width
        )
        
        # Evaluate the image
        scores = evaluate_perspective_image(perspective_img)
        
        # Store results
        results.append({
            'point_name': point_name,
            'target_coords': target_coords,
            'yaw': yaw,
            'pitch': pitch,
            'fov': fov_adjusted,
            'scores': scores,
            'image': perspective_img
        })
        
        logging.info(f"Point {point_name}: combined_score={scores['combined_score']:.3f}, "
                     f"horizontal_edgelets={scores['horizontal_edgelets_count']}")
    
    # Sort results by combined score (descending)
    results.sort(key=lambda x: x['scores']['combined_score'], reverse=True)
    
    # Return the best result
    if results:
        return results[0]
    return None


def visualize_horizontal_edgelets(image, edgelets):
    """Visualize horizontal edgelets on the image."""
    _, directions, strengths = edgelets
    
    if len(directions) == 0:
        return image.copy()
    
    # Create a copy of the image
    vis_image = image.copy()
    
    # Calculate angles and filter horizontal lines
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angles_deg = np.degrees(angles) % 180
    
    # Get locations
    locations, _, _ = edgelets
    
    # Draw horizontal edgelets
    for i, (loc, strength) in enumerate(zip(locations, strengths)):
        # Scale line length by strength
        length = min(strength * 2, 50)
        
        # Calculate endpoint
        angle_rad = angles[i]
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        
        # Draw the line
        start_point = (int(loc[0] - dx/2), int(loc[1] - dy/2))
        end_point = (int(loc[0] + dx/2), int(loc[1] + dy/2))
        
        # Use color to indicate how horizontal the line is
        horizontalness = 1 - min(angles_deg[i], 180 - angles_deg[i]) / 90
        color = (0, int(255 * horizontalness), int(255 * (1 - horizontalness)))
        
        cv2.line(vis_image, start_point, end_point, color, 2)
    
    return vis_image

# Main function to find the best perspective image based on vanishing points
def find_best_perspective_with_vanishing_points(row, images_folder):
    """Find the best perspective image for a building based on vanishing point analysis."""
    camera_coords = (row.loc['X'], row.loc['Y'], row.loc['Z'])
    midpoint_coords = (row.loc['geometry_midpoint'].x, row.loc['geometry_midpoint'].y, row.loc['Z'] + 0.25)

    # Calculate facade extremes
    facade_start = (row.loc['geometry_outer_edge'].coords[0][:2])
    facade_end = (row.loc['geometry_outer_edge'].coords[-1][:2])

    facade_length = np.linalg.norm(np.array([facade_end[0], facade_end[1]]) - np.array([facade_start[0], facade_start[1]]))
    print(f"Facade length: {facade_length}")
    
    target_z = midpoint_coords[2]
    
    # Define the points to evaluate
    points = {
        "midpoint": midpoint_coords,
        "facade_start": np.array([facade_start[0], facade_start[1], target_z]),
        "facade_end": np.array([facade_end[0], facade_end[1], target_z]),
        "projected": np.array([*project_point_onto_segment(camera_coords[:2], facade_start, facade_end), target_z]),
        "mid_mid_start": np.array([(midpoint_coords[0] + facade_start[0]) / 2, (midpoint_coords[1] + facade_start[1]) / 2, target_z]),
        "mid_mid_end": np.array([(midpoint_coords[0] + facade_end[0]) / 2, (midpoint_coords[1] + facade_end[1]) / 2, target_z])
    }
    
    # Calculate distance to check if the facade is too far
    x1, y1, _ = camera_coords
    x2, y2, _ = midpoint_coords
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # If too far, skip this building
    if distance > 15:
        print("Facade too far from camera, probably part of block not facing street")
        return None
    
    # Load the panorama image
    path = row.loc['PATH']
    run = row.loc['RUN']
    image_name = row.loc['FOTO']
    image_path = '{}{}/{}/{}'.format(images_folder, path, run, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
        
    rgb_image = convert_bgr_to_rgb(image)
    roll = row.loc['Roll']
    
    # Evaluate all target points
    best_result = evaluate_perspective_images_for_points(
        rgb_image, camera_coords, points, target_z, roll
    )
    
    if best_result is None:
        return None
    
    # Apply pitch adjustment
    best_pitch = calculate_pitch_adjustment_based_on_distance(
        camera_coords, best_result['target_coords'], best_result['pitch']
    )
    
    # Generate the final perspective image with adjusted pitch
    final_image = enhanced_equirectangular_to_perspective(
        rgb_image, best_result['fov'], best_result['yaw'], best_pitch, roll, 1024, 1024
    )
    
    # Compute edgelets for visualization
    edgelets = compute_edgelets(final_image, sigma=1.5)
    horizontal_edgelets = filter_edgelets_by_orientation(edgelets, orientation='horizontal', threshold=20)
    
    # Visualize the horizontal edgelets
    vis_image = visualize_horizontal_edgelets(final_image, horizontal_edgelets)
    
    return {
        'image': final_image,
        'visualization': vis_image,
        'target_coords': best_result['target_coords'],
        'yaw': best_result['yaw'],
        'pitch': best_pitch,
        'fov': best_result['fov'],
        'point_name': best_result['point_name'],
        'scores': best_result['scores']
    }

# Main loop to process all buildings
plt.close('all')

images_folder = 'data/PANO_new/'
output_folder = 'data/Street_view/Output_images_vp/'
visualization_folder = 'data/Street_view/Output_visualization/'

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# Dictionary to track how many images we've saved per building
building_image_count = {}

# Create a new column for vanishing point scores
if 'vp_scores' not in gdf_combined.columns:
    gdf_combined['vp_scores'] = None

for i, row in gdf_combined.iterrows():
    print(f"\nProcessing building {i+1}/{len(gdf_combined)}")
    
    # Find the best perspective image
    result = find_best_perspective_with_vanishing_points(row, images_folder)
    
    if result is None:
        print(f"Skipping building {row.loc['building_id']}")
        continue
    
    # Handle filename logic
    building_id = str(row.loc['building_id'])
    if building_id not in building_image_count:
        building_image_count[building_id] = 0
    
    building_image_count[building_id] += 1
    suffix = ''
    
    # If more than one image for the building, append a letter (A, B, C, etc.)
    if building_image_count[building_id] > 1:
        suffix = string.ascii_uppercase[building_image_count[building_id] - 2]

    # Create the filenames
    output_filename = f"Building_{building_id}{suffix}.png"
    output_path = os.path.join(output_folder, output_filename)
    
    vis_filename = f"Building_{building_id}{suffix}_vis.png"
    vis_path = os.path.join(visualization_folder, vis_filename)

    # Save the images
    plt.imsave(output_path, result['image'])
    plt.imsave(vis_path, result['visualization'])

    # Add the output filename and scores to the GeoDataFrame
    gdf_combined.at[i, 'output_filename'] = output_filename
    gdf_combined.at[i, 'vp_scores'] = str(result['scores'])

    # Display results
    print(f"Building {building_id}: Best point is {result['point_name']}")
    print(f"Scores: parallelism={result['scores']['parallelism_score']:.3f}, "
          f"alignment={result['scores']['alignment_score']:.3f}, "
          f"horizontal_lines={result['scores']['horizontal_edgelets_count']}")
    
    # Plot the result
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(result['image'])
    axes[0].set_title(f"Best Perspective (FOV={result['fov']:.1f}°)")
    axes[0].axis('off')
    
    axes[1].imshow(result['visualization'])
    axes[1].set_title(f"Horizontal Edgelets Score: {result['scores']['combined_score']:.3f}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_folder, f"Building_{building_id}{suffix}_plot.png"))
    plt.close('all')

# Drop unwanted geometry columns before saving
gdf_combined = gdf_combined.drop(columns=['geometry_midpoint'])

# Set 'geometry' as the active geometry column
gdf_combined = gdf_combined.set_geometry('geometry')

# Save the GeoDataFrame to a file
gdf_combined.to_file('data/Street_view/Preprocessed_with_vp_filenames.geojson', driver='GeoJSON')
