import os
openslide_bin_path = r'C:\Users\Jingy\OneDrive\Desktop\nuclei_segmentation\openslide-bin-4.0.0.2-windows-x64\bin'
os.add_dll_directory(openslide_bin_path)
import openslide
from openslide import open_slide
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import os

# Function to read XML annotations and create masks for glomeruli
def create_glomeruli_masks(wsi_path, xml_path, output_folder):
    # Load the whole slide image
    slide = open_slide(wsi_path)
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Process each annotation to create masks
    for i, annotation in enumerate(root.findall('.//Annotation')):
        # Find all coordinates within each Annotation
        regions = annotation.findall('.//Region')
        for j, region in enumerate(regions):
            vertices = region.findall('.//Vertex')
            # Get the coordinates for the region
            points = [(int(float(vertex.get('X'))), int(float(vertex.get('Y')))) for vertex in vertices]

            # Determine the bounding box for the region
            x_coords, y_coords = zip(*points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min

            # Create a mask for the bounding box
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Offset points for mask polygon
            offset_points = [(x - x_min, y - y_min) for x, y in points]
            cv2.fillPoly(mask, [np.array(offset_points)], 255)

            # Save mask image
            mask_image = Image.fromarray(mask)
            mask_image.save(f'{output_folder}/glomerulus_mask_{i}_{j}.png')

    print(f"Extracted {i+1} annotations with total {j+1} regions.")

# Main program entry
if __name__ == "__main__":
    # Specify the WSI file path, XML annotation file path, and output folder
    wsi_file_path = 'P44F_PAS.svs'
    xml_file_path = 'P44F_PAS.xml'
    output_dir = 'data/glomeruli_masks'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the function to create masks based on annotations
    create_glomeruli_masks(wsi_file_path, xml_file_path, output_dir)
