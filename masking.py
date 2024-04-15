import os
import cv2
import numpy as np
from skimage import io, filters, exposure, feature, morphology
from skimage.color import rgb2gray
from skimage.morphology import disk, binary_closing, binary_opening, binary_dilation, binary_erosion
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import watershed

# Function to apply Gabor filter
def apply_gabor_filter(image, frequency, theta):
    gabor_kernel = cv2.getGaborKernel((21, 21), 3, theta, frequency, 0.2, 0, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

# Function to enhance contrast using histogram equalization
def enhance_contrast(image):
    return exposure.equalize_adapthist(image, clip_limit=0.03)

# Function to generate glomerulus mask
def generate_glomerulus_mask(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = enhance_contrast(image)
    num_angles = 8
    gabor_responses = []
    for theta in np.linspace(0, np.pi, num_angles, endpoint=False):
        gabor_response = apply_gabor_filter(image, 0.2, theta)
        gabor_responses.append(gabor_response)
    magnitude = np.maximum.reduce(gabor_responses)
    
    edges = feature.canny(magnitude, sigma=2)
    closed_edges = binary_closing(edges, disk(3))
    thresh = filters.threshold_otsu(magnitude)
    binary_mask = magnitude > thresh
    binary_mask = binary_mask | closed_edges
    for size in [1, 2, 3]: 
        selem = disk(size)
        binary_mask = binary_closing(binary_mask, selem)
        binary_mask = binary_opening(binary_mask, selem)
    filled_mask = ndi.binary_fill_holes(binary_mask)
    clean_mask = morphology.remove_small_objects(filled_mask, min_size=150)
    clean_mask = morphology.remove_small_holes(clean_mask, area_threshold=150)
    return (clean_mask * 255).astype(np.uint8)

# Function to isolate largest component
def isolate_largest_component(mask):
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask
    max_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    largest_component = np.zeros_like(mask)
    largest_component[labels_im == max_label] = 255
    return largest_component

# Function to refine mask
def process_mask(mask_path, output_dir):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    refined_mask = isolate_largest_component(mask)
    refined_mask = morphology.binary_dilation(refined_mask, disk(2))
    final_mask = np.zeros_like(mask)
    final_mask[refined_mask > 0] = 255
    base_filename = os.path.basename(mask_path).split('.')[0] + '_final_mask.png'
    final_mask_path = os.path.join(output_dir, base_filename)
    io.imsave(final_mask_path, final_mask)
    print(f"Final mask saved to {final_mask_path}")

# Function to process a folder of images
def process_folder(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif'):
            image_path = os.path.join(input_dir, filename)
            mask = generate_glomerulus_mask(image_path)
            mask_filename = filename.split('.')[0] + '_boundary_mask.png'
            mask_path = os.path.join(output_dir, mask_filename)
            io.imsave(mask_path, mask)
            print(f"Mask saved for {filename} to {mask_path}")

# Paths to the folders
input_dir = 'data/original_images'
boundary_mask_dir = 'data/boundary_masks'
final_mask_dir = 'data/final_masks'

# Generate boundary masks
process_folder(input_dir, boundary_mask_dir)

# Refine masks to get the final masks
for mask_filename in os.listdir(boundary_mask_dir):
    if mask_filename.lower().endswith('_boundary_mask.png'):
        mask_path = os.path.join(boundary_mask_dir, mask_filename)
        process_mask(mask_path, final_mask_dir)
