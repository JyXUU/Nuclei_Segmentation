import json
import numpy as np
import cv2
from shapely.geometry import shape, Polygon
from skimage import io, filters, morphology
import os
import matplotlib.pyplot as plt

# Base directory for data
BASE_DIR = 'data'

# Directories containing the GeoJSON and segmented image files
GEOJSON_DIR = os.path.join(BASE_DIR, 'gt')
SEGMENTED_IMAGES_DIR = os.path.join(BASE_DIR, 'segment')

def parse_geojson(geojson_file):
    with open(geojson_file, 'r') as file:
        data = json.load(file)
    annotations = [shape(feature["geometry"]).buffer(0) for feature in data["features"]]
    return annotations

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

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def process_image(segmented_image, sigma, min_size, area_threshold):
    # Gaussian blur to smooth the image
    blurred_image = filters.gaussian(segmented_image, sigma=sigma)
    
    # Convert image to binary using Otsu's method
    thresh = filters.threshold_otsu(blurred_image)
    binary_mask = blurred_image > thresh

    # Morphological operations
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=area_threshold)

    return cleaned_mask.astype(np.uint8)

# Fixed parameters
fixed_sigma = 1.0
fixed_min_size = 40
fixed_area_threshold = 40

# Varying parameters
sigmas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
min_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
area_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Results dictionaries
results_sigma = {sigma: [] for sigma in sigmas}
results_min_size = {min_size: [] for min_size in min_sizes}
results_area_threshold = {area_threshold: [] for area_threshold in area_thresholds}

for geojson_file in sorted(os.listdir(GEOJSON_DIR)):
    if geojson_file.endswith('.geojson'):
        geojson_path = os.path.join(GEOJSON_DIR, geojson_file)
        annotations = parse_geojson(geojson_path)
        
        base_name = geojson_file.split('.')[0]
        image_file_name = f'{base_name}_07_cleaned_distance.jpg'
        segmented_image_path = os.path.join(SEGMENTED_IMAGES_DIR, image_file_name)
        segmented_image = io.imread(segmented_image_path, as_gray=True)
        
        image_shape = segmented_image.shape
        ground_truth_mask = create_mask_from_polygons(annotations, image_shape)
        
        # Sigma sensitivity
        for sigma in sigmas:
            processed_mask = process_image(segmented_image, sigma, fixed_min_size, fixed_area_threshold)
            iou_score = calculate_iou(ground_truth_mask, processed_mask)
            results_sigma[sigma].append(iou_score)

        # Min size sensitivity
        for min_size in min_sizes:
            processed_mask = process_image(segmented_image, fixed_sigma, min_size, fixed_area_threshold)
            iou_score = calculate_iou(ground_truth_mask, processed_mask)
            results_min_size[min_size].append(iou_score)

        # Area threshold sensitivity
        for area_threshold in area_thresholds:
            processed_mask = process_image(segmented_image, fixed_sigma, fixed_min_size, area_threshold)
            iou_score = calculate_iou(ground_truth_mask, processed_mask)
            results_area_threshold[area_threshold].append(iou_score)

# Plotting results
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Sigma plot
for sigma, scores in results_sigma.items():
    axs[0].plot(sigmas, [np.mean(results_sigma[sigma]) for sigma in sigmas], label=f'Sigma={sigma}')
axs[0].set_title('IoU by Sigma')
axs[0].set_xlabel('Sigma')
axs[0].set_ylabel('Average IoU Score')

# Min Size plot
for min_size, scores in results_min_size.items():
    axs[1].plot(min_sizes, [np.mean(results_min_size[min_size]) for min_size in min_sizes], label=f'Min Size={min_size}')
axs[1].set_title('IoU by Min Size')
axs[1].set_xlabel('Min Size')
axs[1].set_ylabel('Average IoU Score')

# Area Threshold plot
for area_threshold, scores in results_area_threshold.items():
    axs[2].plot(area_thresholds, [np.mean(results_area_threshold[area_threshold]) for area_threshold in area_thresholds], label=f'Area Threshold={area_threshold}')
axs[2].set_title('IoU by Area Threshold')
axs[2].set_xlabel('Area Threshold')
axs[2].set_ylabel('Average IoU Score')

plt.tight_layout()
plt.show()
