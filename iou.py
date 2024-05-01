import json
import numpy as np
import cv2
from shapely.geometry import shape, Polygon
from skimage import io
import os

# Base directory for data
BASE_DIR = 'data'

# Directories containing the GeoJSON and segmented image files
GEOJSON_DIR = os.path.join(BASE_DIR, 'Gloms_segmented_annotation')
SEGMENTED_IMAGES_DIR = os.path.join(BASE_DIR, 'Nuclei_segmented')
GROUND_TRUTH_IMAGES_DIR = os.path.join(BASE_DIR, 'Gloms_segmented_img')

# Create the directory for ground truth images if it does not exist
if not os.path.exists(GROUND_TRUTH_IMAGES_DIR):
    os.makedirs(GROUND_TRUTH_IMAGES_DIR)

# Function to parse the GeoJSON file and return the annotations as polygons
def parse_geojson(geojson_file):
    with open(geojson_file, 'r') as file:
        data = json.load(file)
    annotations = [shape(feature["geometry"]).buffer(0) for feature in data["features"]]
    return annotations

# Function to create binary masks from polygons
def create_mask_from_polygons(polygons, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for poly in polygons:
        if isinstance(poly, Polygon):
            exterior = np.array(poly.exterior.coords).round().astype(np.int32)
            cv2.fillPoly(mask, [exterior], 255)
        else:
            for geom in poly.geoms:
                exterior = np.array(geom.exterior.coords).round().astype(np.int32)
                cv2.fillPoly(mask, [exterior], 255)
    return mask

# Function to calculate IoU
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# List of GeoJSON files sorted numerically
geojson_files = sorted(
    [f for f in os.listdir(GEOJSON_DIR) if f.endswith('.geojson')],
    key=lambda x: int(x.split('_')[1])
)

# Process each GeoJSON file and calculate IoU
iou_scores = []
for geojson_file in geojson_files:
    geojson_path = os.path.join(GEOJSON_DIR, geojson_file)
    annotations = parse_geojson(geojson_path)

    # Construct the image file name based on the GeoJSON file name
    base_name = geojson_file.split('.')[0]
    image_file_name = f'{base_name}_06_cleaned_distance.jpg'
    segmented_image_path = os.path.join(SEGMENTED_IMAGES_DIR, image_file_name)

    # Load the corresponding segmented image
    segmented_image = io.imread(segmented_image_path, as_gray=True)
    if len(segmented_image.shape) > 2:
        segmented_image = segmented_image[:, :, 0]  # Use the first channel if RGB

    # Ensure it is a binary mask
    segmented_mask = segmented_image > 127  # Thresholding, adjust value as needed

    # Image dimensions from the segmented image
    image_shape = segmented_mask.shape

    # Create binary masks for all annotations
    ground_truth_mask = create_mask_from_polygons(annotations, image_shape)

    # Save the ground truth mask as an image
    ground_truth_image_path = os.path.join(GROUND_TRUTH_IMAGES_DIR, f"{base_name}_mask.png")
    io.imsave(ground_truth_image_path, ground_truth_mask.astype(np.uint8))

    # Calculate IoU for the current image
    iou_score = calculate_iou(ground_truth_mask, segmented_mask)
    iou_scores.append(iou_score)

average_iou = np.mean(iou_scores)
print(f'Average IoU score for the dataset is: {average_iou}')


# Average IoU score for the dataset is: 0.20411747320401266