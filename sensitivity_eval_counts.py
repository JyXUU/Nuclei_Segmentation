import os
import numpy as np
import cv2
from skimage import io
from skimage.color import rgb2hed
from skimage.filters import gaussian, threshold_otsu
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
import matplotlib.pyplot as plt

def save_image(image, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if image.dtype != np.uint8:
        if np.max(image) > 1:
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, filename), image)

def apply_morphological_operations(mask, min_size, area_threshold):
    struct_elem = disk(3)
    mask_opened = opening(mask, struct_elem)
    mask_cleaned = closing(mask_opened, struct_elem)
    mask_cleaned = remove_small_objects(mask_cleaned.astype(bool), min_size=min_size)
    mask_cleaned = remove_small_holes(mask_cleaned, area_threshold=area_threshold)
    return (mask_cleaned * 255).astype(np.uint8)

# Include your image processing functions here.

def process_folder(data_folder, masks_folder, output_folder, step_output_folder, sigma, min_size, area_threshold):
    nuclei_counts = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.png') and 'glomerulus' in filename:
            image_path = os.path.join(data_folder, filename)
            mask_image_filename = f"{os.path.splitext(filename)[0].replace('glomerulus', 'glomerulus_mask')}.png"
            mask_image_path = os.path.join(masks_folder, mask_image_filename)
            base_filename = os.path.splitext(filename)[0]
            if os.path.exists(mask_image_path):
                nuclei_counts[base_filename] = process_image(image_path, mask_image_path, output_folder, step_output_folder, sigma, min_size, area_threshold)
            else:
                print(f"Mask image not found for {filename}")
    return nuclei_counts

def process_image(image_path, mask_image_path, output_dir, step_output_dir, sigma, min_size, area_threshold):
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save the original image
    save_image(image_rgb, step_output_dir, f"{base_filename}_01_original.jpg")

    # Read the mask and save it
    final_mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    save_image(final_mask, step_output_dir, f"{base_filename}_02_final_mask.jpg")

    # Convert to HED and isolate the hematoxylin channel
    hed = rgb2hed(image_rgb)
    hematoxylin_channel = hed[:, :, 0]

    # Apply Gaussian filtering with the given sigma
    blurred = gaussian(hematoxylin_channel, sigma=sigma)
    thresh = threshold_otsu(blurred)
    binary_nuclei = (blurred > thresh).astype(np.uint8) * 255
    save_image(binary_nuclei, step_output_dir, f"{base_filename}_03_binary_nuclei.jpg")

    # Perform morphological operations
    cleaned_mask = apply_morphological_operations(binary_nuclei, min_size, area_threshold)
    save_image(cleaned_mask, step_output_dir, f"{base_filename}_06_cleaned_distance.jpg")

    # Perform watershed segmentation
    distance = distance_transform_edt(cleaned_mask)
    markers = label(distance > 0)
    labels = watershed(-distance, markers, mask=binary_nuclei)

    # Save final labeled image
    save_image(labels, step_output_dir, f"{base_filename}_07_labels.jpg")

    # Count nuclei
    nuclei_count = len(np.unique(labels[final_mask > 0])) - 1
    print(f"Nuclei count in {base_filename} within glom area: {nuclei_count}")

    return nuclei_count


def read_count_file(filepath):
    counts = {}
    with open(filepath, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            counts[key.strip()] = int(value.strip())
    return counts

def calculate_image_accuracy(actual_count, manual_count):
    return (actual_count / manual_count) * 100 if manual_count > 0 else 0

def calculate_overall_accuracy(predicted_counts, manual_counts):
    accuracies = []
    for image_name, manual_count in manual_counts.items():
        actual_count = predicted_counts.get(image_name, 0)
        accuracies.append(calculate_image_accuracy(actual_count, manual_count))
    return np.mean(accuracies), accuracies


def plot_parameter(ax, parameter_values, accuracies, parameter_name):
    ax.plot(parameter_values, accuracies, marker='', linestyle='-')
    ax.set_title(f'Accuracy by {parameter_name}')
    ax.set_xlabel(parameter_name)
    ax.set_ylabel('Accuracy (%)')


def main():
    sigma_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    min_size_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    area_threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    manual_counts = read_count_file('manual_counts.txt')
    actual_counts = read_count_file(r'results\segmentation_results\nuclei_counts.txt')

    data_folder = 'data/output_glomeruli'
    masks_folder = 'data/glomeruli_masks'
    output_folder = 'results/segmentation_results'
    step_output_folder = 'results/step_results'

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Vary sigma while keeping min_size and area_threshold constant
    constant_min_size = 40
    constant_area_threshold = 40
    sigma_accuracies = []
    for sigma in sigma_values:
        nuclei_counts = process_folder(data_folder, masks_folder, output_folder, step_output_folder, sigma, constant_min_size, constant_area_threshold)
        overall_accuracy, _ = calculate_overall_accuracy(nuclei_counts, manual_counts)
        sigma_accuracies.append(overall_accuracy)
    plot_parameter(axes[0], sigma_values, sigma_accuracies, "Sigma")

    # Vary min_size while keeping sigma and area_threshold constant
    constant_sigma = 1
    min_size_accuracies = []
    for min_size in min_size_values:
        nuclei_counts = process_folder(data_folder, masks_folder, output_folder, step_output_folder, constant_sigma, min_size, constant_area_threshold)
        overall_accuracy, _ = calculate_overall_accuracy(nuclei_counts, manual_counts)
        min_size_accuracies.append(overall_accuracy)
    plot_parameter(axes[1], min_size_values, min_size_accuracies, "Min Size")

    # Vary area_threshold while keeping sigma and min_size constant
    area_threshold_accuracies = []
    for area_threshold in area_threshold_values:
        nuclei_counts = process_folder(data_folder, masks_folder, output_folder, step_output_folder, constant_sigma, constant_min_size, area_threshold)
        overall_accuracy, _ = calculate_overall_accuracy(nuclei_counts, manual_counts)
        area_threshold_accuracies.append(overall_accuracy)
    plot_parameter(axes[2], area_threshold_values, area_threshold_accuracies, "Area Threshold")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
