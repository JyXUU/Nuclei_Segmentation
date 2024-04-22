import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import io
from skimage.color import label2rgb, rgb2gray
from math import log2, sqrt, floor, pi

# Function to calculate the range of wavelengths for Gabor filters based on the image size
def calculate_wavelengths(image_size):
    numRows, numCols = image_size
    wavelengthMin = 4 / sqrt(2)
    wavelengthMax = sqrt(numRows**2 + numCols**2)
    n = floor(log2(wavelengthMax / wavelengthMin))
    return 2**np.arange(0, n-2) * wavelengthMin

# Function to apply Gabor filters to an image using specified wavelengths and orientations
def apply_gabor_filters(img, wavelengths, orientations):
    accum = np.zeros_like(img, dtype=np.float32)
    for wavelength in wavelengths:
        for theta in orientations:
            kernel = cv2.getGaborKernel(ksize=(int(3 * wavelength), int(3 * wavelength)),
                                        sigma=wavelength / pi,
                                        theta=theta,
                                        lambd=wavelength,
                                        gamma=1,
                                        psi=0)
            filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
            np.maximum(accum, filtered, accum)
    accum = cv2.normalize(accum, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return accum

# Function to perform image processing and glomeruli segmentation
def process_image(file_path, min_glom_size):
    # Read image
    img = io.imread(file_path)
    # Convert image to grayscale
    Agray = rgb2gray(img)
    # Get image size
    image_size = Agray.shape

    # Define Gabor filter parameters
    wavelengths = calculate_wavelengths(image_size)
    orientations = np.deg2rad(np.arange(0, 180, 15))  # Adjusted orientation range

    # Apply Gabor filters
    gabormag = apply_gabor_filters(Agray, wavelengths, orientations)

    # Flatten the filtered image and normalize
    X = gabormag.reshape(-1, 1)
    X = (X - np.mean(X)) / np.std(X)

    # Perform PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)
    feature2DImage = X_pca.reshape(image_size)

    # Perform k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, n_init=5)  # Adjusted number of clusters
    L = kmeans.fit_predict(X_pca)
    L = L.reshape(image_size)  

    # Find the label corresponding to gloms
    glom_label = np.argmax(np.bincount(L.flatten()))

    # Create label image
    label_image = label2rgb(L, image=img, bg_label=0)

    # Use the label to mask the image
    mask = L == glom_label

    # Find connected components and filter by size
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    for label_id, stat in enumerate(stats[1:], start=1):
        # Ignore small objects
        if stat[4] < min_glom_size:
            mask[labels == label_id] = False

    # Update segmented image with the filtered mask
    segmented_image = np.zeros_like(img)
    for i in range(3):  # Apply the updated mask to each of the color channels
        segmented_image[:, :, i] = np.where(mask, img[:, :, i], 0)

    return feature2DImage, label_image, segmented_image

# Function to normalize an image to the 0-255 range and convert to uint8
def normalize_and_convert_to_uint8(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_image.astype(np.uint8)

# Function to process all images in a directory
def process_images(directory, min_glom_size):
    output_dir = os.path.join(directory, 'processed_images')
    os.makedirs(output_dir, exist_ok=True)

    file_list = [file for file in os.listdir(directory) if file.lower().endswith('.tif')]

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        feature2DImage, label_image, segmented_image = process_image(file_path, min_glom_size)  

        basename = os.path.splitext(file_name)[0]
        # Normalize and convert feature2DImage before saving
        feature2DImage_uint8 = normalize_and_convert_to_uint8(feature2DImage)
        io.imsave(os.path.join(output_dir, f'{basename}_feature2DImage.png'), feature2DImage_uint8)
        # Convert label image to uint8 before saving
        label_image_uint8 = (label_image * 255).astype(np.uint8)
        io.imsave(os.path.join(output_dir, f'{basename}_label_image.png'), label_image_uint8)
        # Normalize and convert segmented image before saving
        segmented_image_uint8 = normalize_and_convert_to_uint8(segmented_image)
        io.imsave(os.path.join(output_dir, f'{basename}_segmented_image.png'), segmented_image_uint8)
        print(f'Processed and saved results for {file_name}')

if __name__ == '__main__':
    # Replace 'windows' with the path to your image directory
    min_glom_size = 10000  # Adjust the minimum glom size as needed
    process_images('windows', min_glom_size)
