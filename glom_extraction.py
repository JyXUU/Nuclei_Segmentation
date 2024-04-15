import os
openslide_bin_path = r'C:\Users\Jingy\OneDrive\Desktop\nuclei_segmentation\openslide-bin-4.0.0.2-windows-x64\bin'
os.add_dll_directory(openslide_bin_path)
import openslide
from openslide import open_slide
import numpy as np
import cv2
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image

# Function to read XML annotations and extract glomeruli
def extract_glomeruli_from_annotations(wsi_path, xml_path, output_folder):
    # Load the whole slide image
    slide = open_slide(wsi_path)
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract annotations and corresponding regions
    for i, annotation in enumerate(root.findall('.//Annotation')):
        # Find all coordinates within each Annotation
        regions = annotation.findall('.//Region')
        for j, region in enumerate(regions):
            vertices = region.findall('.//Vertex')
            # Get the coordinates for the region
            x_coords = [int(float(vertex.get('X'))) for vertex in vertices]
            y_coords = [int(float(vertex.get('Y'))) for vertex in vertices]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min

            # Extract region of interest from the original slide
            roi = slide.read_region((x_min, y_min), 0, (width, height))
            roi = roi.convert("RGB")  # Convert from RGBA to RGB
            roi.save(f'{output_folder}/glomerulus_{i}_{j}.png')

    print(f"Extracted {i+1} annotations with total {j+1} regions.")

# Main program entry
if __name__ == "__main__":
    # Specify the WSI file path, XML annotation file path, and output folder
    wsi_file_path = 'P44F_PAS.svs'
    xml_file_path = 'P44F_PAS.xml'
    output_dir = 'data/output_glomeruli'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run the function to extract regions based on annotations
    extract_glomeruli_from_annotations(wsi_file_path, xml_file_path, output_dir)

