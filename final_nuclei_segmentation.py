import os
import cv2
import numpy as np
from skimage import io, exposure
from skimage.color import rgb2hed, hed2rgb
from skimage.filters import gaussian, threshold_otsu, threshold_triangle, threshold_yen
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes, erosion

def save_image(image, output_dir, filename, is_normalized=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if image.dtype != np.uint8 and is_normalized:
        image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
    io.imsave(os.path.join(output_dir, filename), image)

def apply_morphological_operations(mask):
    struct_elem = disk(3)
    mask_opened = opening(mask, struct_elem)
    mask_cleaned = closing(mask_opened, struct_elem)
    mask_eroded = erosion(mask_cleaned, struct_elem)  # Erode to improve marker separation
    mask_cleaned_final = remove_small_objects(mask_eroded.astype(bool), min_size=100)
    mask_cleaned_final = remove_small_holes(mask_cleaned_final, area_threshold=60)
    return (mask_cleaned_final * 255).astype(np.uint8)

def process_image(image_path, output_dir, step_output_dir):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_image(image_rgb, step_output_dir, f"{base_filename}_01_original.jpg", False)

    # Mask creation
    final_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final_mask = (final_mask > 10).astype(np.uint8) * 255
    save_image(final_mask, step_output_dir, f"{base_filename}_02_final_mask.jpg")

    # Hematoxylin channel extraction and inversion
    hed = rgb2hed(image_rgb)
    hematoxylin_channel = hed[:, :, 0]
    rescaled_hematoxylin = exposure.rescale_intensity(hematoxylin_channel, in_range=(0, np.max(hematoxylin_channel)), out_range=(1, 0))
    save_image(rescaled_hematoxylin, step_output_dir, f"{base_filename}_02b_hematoxylin.jpg")

    # Adaptive histogram equalization
    equalized_hematoxylin = exposure.equalize_adapthist(rescaled_hematoxylin, clip_limit=0.9)
    save_image(equalized_hematoxylin, step_output_dir, f"{base_filename}_02c_equalized_hematoxylin.jpg")

    # Gaussian blur and thresholding
    blurred = gaussian(equalized_hematoxylin, sigma=1)
    thresh = threshold_yen(blurred)
    binary_nuclei = (blurred > thresh).astype(np.uint8) * 255
    # This binary image should have nuclei as white and background as black, so invert if necessary
    binary_nuclei = np.where(binary_nuclei == 255, 0, 255)
    save_image(binary_nuclei, step_output_dir, f"{base_filename}_03_binary_nuclei.jpg")

    # Distance transform uses the binary nuclei where nuclei are meant to be white
    distance = distance_transform_edt(binary_nuclei == 0)
    save_image(distance, step_output_dir, f"{base_filename}_04_distance.jpg")

    cleaned_distance = apply_morphological_operations(binary_nuclei)
    save_image(cleaned_distance, nuclei_mask_folder, f"{base_filename}_06_cleaned_distance.jpg")

    markers = label(cleaned_distance > 0)
    mask_for_watershed = (cleaned_distance == 255)  # Use the cleaned mask for the watershed

    labels = watershed(-distance_transform_edt(mask_for_watershed), markers, mask=mask_for_watershed)
    save_image(labels, step_output_dir, f"{base_filename}_07_labels.jpg")

    labels_in_glom_only = labels * (final_mask // 255)
    image_overlay = image_rgb.copy()
    for label_id in np.unique(labels_in_glom_only):
        if label_id == 0:
            continue
        nucleus_mask = labels_in_glom_only == label_id
        image_overlay[nucleus_mask] = [255, 0, 0]
    save_image(image_overlay, step_output_dir, f"{base_filename}_08_overlay_nuclei.jpg")

    io.imsave(os.path.join(output_dir, f"{base_filename}_overlay_nuclei.jpg"), image_overlay)

    nuclei_count = len(np.unique(labels_in_glom_only)) - 1
    print(f"Nuclei count in {base_filename} within glom area: {nuclei_count}")
    return nuclei_count

def process_folder(data_folder, output_folder, step_output_folder):
    nuclei_counts = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.tif'):
            image_path = os.path.join(data_folder, filename)
            base_filename = os.path.splitext(filename)[0]
            nuclei_counts[base_filename] = process_image(image_path, output_folder, step_output_folder)

    counts_filename = os.path.join(output_folder, 'nuclei_counts.txt')
    with open(counts_filename, 'w') as f:
        for base_filename, count in nuclei_counts.items():
            f.write(f"{base_filename}: {count}\n")
    print(f"Nuclei counts saved to {counts_filename}")

# Paths to the folders
data_folder = 'data/Gloms_segmented'
output_folder = 'final_results/segmentation_results'
step_output_dir = 'final_results/step_results'
nuclei_mask_folder = 'data/Nuclei_segmented'

process_folder(data_folder, output_folder, step_output_dir)
