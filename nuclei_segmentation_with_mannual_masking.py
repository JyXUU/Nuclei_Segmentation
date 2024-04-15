import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from skimage import io
from skimage.color import rgb2hed
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes

def save_image(image, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if image.dtype != np.uint8:
        if np.max(image) > 1:
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, filename), image)

def apply_morphological_operations(mask):
    struct_elem = disk(3)
    mask_opened = opening(mask, struct_elem)
    mask_cleaned = closing(mask_opened, struct_elem)
    mask_cleaned = remove_small_objects(mask_cleaned.astype(bool), min_size=40)
    mask_cleaned = remove_small_holes(mask_cleaned, area_threshold=40)
    return (mask_cleaned * 255).astype(np.uint8)

def parse_xml_boundary_mask(xml_path, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for region in root.findall('.//Region'):
        vertices = []
        for vertex in region.findall('.//Vertex'):
            x = int(float(vertex.get('X')))
            y = int(float(vertex.get('Y')))
            vertices.append((x, y))
        cv2.fillPoly(mask, [np.array(vertices, np.int32)], color=255)
    return mask

def process_image(image_path, mask_image_path, output_dir, step_output_dir):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    save_image(image_rgb, step_output_dir, f"{base_filename}_01_original.jpg")
    
    final_mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    save_image(final_mask, step_output_dir, f"{base_filename}_02_final_mask.jpg")

    hed = rgb2hed(image_rgb)
    hematoxylin_channel = hed[:, :, 0]

    blurred = gaussian(hematoxylin_channel, sigma=1)
    thresh = threshold_otsu(blurred)
    binary_nuclei = (blurred > thresh).astype(np.uint8) * 255
    save_image(binary_nuclei, step_output_dir, f"{base_filename}_03_binary_nuclei.jpg")

    distance = distance_transform_edt(binary_nuclei)
    save_image(distance, step_output_dir, f"{base_filename}_04_distance.jpg")

    distance_in_glom = np.where(final_mask, distance, 0)
    save_image(distance_in_glom, step_output_dir, f"{base_filename}_05_distance_in_glom.jpg")

    cleaned_distance = apply_morphological_operations(distance_in_glom)
    save_image(cleaned_distance, step_output_dir, f"{base_filename}_06_cleaned_distance.jpg")

    markers = label(cleaned_distance > 0)
    labels = watershed(-cleaned_distance, markers, mask=binary_nuclei)
    save_image(labels, step_output_dir, f"{base_filename}_07_labels.jpg")

    labels_in_glom_only = labels * (final_mask > 0)
    image_overlay = image_rgb.copy()
    for label_id in np.unique(labels_in_glom_only):
        if label_id == 0:
            continue
        nucleus_mask = labels_in_glom_only == label_id
        image_overlay[nucleus_mask] = [255, 0, 0]
    save_image(image_overlay, step_output_dir, f"{base_filename}_08_overlay_nuclei.jpg")

    io.imsave(os.path.join(output_dir, f"{base_filename}_overlay_nuclei.jpg"), image_overlay)

    nuclei_count = len(np.unique(labels[final_mask > 0])) - 1
    print(f"Nuclei count in {base_filename} within glom area: {nuclei_count}")
    return nuclei_count

def process_folder(data_folder, masks_folder, output_folder, step_output_folder):
    nuclei_counts = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.png') and 'glomerulus' in filename:
            image_path = os.path.join(data_folder, filename)
            # Construct mask filename based on the expected naming convention
            mask_image_filename = f"{os.path.splitext(filename)[0].replace('glomerulus', 'glomerulus_mask')}.png"
            mask_image_path = os.path.join(masks_folder, mask_image_filename)
            base_filename = os.path.splitext(filename)[0]
            if os.path.exists(mask_image_path):
                nuclei_counts[base_filename] = process_image(image_path, mask_image_path, output_folder, step_output_folder)
            else:
                print(f"Mask image not found for {filename}")

    counts_filename = os.path.join(output_folder, 'nuclei_counts.txt')
    with open(counts_filename, 'w') as f:
        for base_filename, count in nuclei_counts.items():
            f.write(f"{base_filename}: {count}\n")
    print(f"Nuclei counts saved to {counts_filename}")

# Paths to the folders
data_folder = 'data/output_glomeruli'
masks_folder = 'data/glomeruli_masks'
output_folder = 'results/segmentation_results'
step_output_folder = 'results/step_results'

process_folder(data_folder, masks_folder, output_folder, step_output_folder)
