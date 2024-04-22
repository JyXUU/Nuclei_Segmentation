import os
openslide_bin_path = r'C:\Users\Jingy\OneDrive\Desktop\nuclei_segmentation\openslide-bin-4.0.0.2-windows-x64\bin'
os.add_dll_directory(openslide_bin_path)
import openslide
from PIL import Image
import numpy as np

def save_windows_from_wsi(slide_path, output_dir, window_size, stride, scale_factor=0.5):
    slide = openslide.OpenSlide(slide_path)
    level = 0  # Using the highest resolution level
    dims = slide.level_dimensions[level]

    os.makedirs(output_dir, exist_ok=True)
    
    for y in range(0, dims[1], stride):
        for x in range(0, dims[0], stride):
            if x + window_size > dims[0] or y + window_size > dims[1]:
                continue  # Skip if the window exceeds the image dimensions
            window = slide.read_region((x, y), level, (window_size, window_size))
            window = np.array(window)[:, :, :3]  # Convert RGBA to RGB
            img = Image.fromarray(window)
            if scale_factor != 1:
                new_size = (int(window_size * scale_factor), int(window_size * scale_factor))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(os.path.join(output_dir, f'window_{x}_{y}.tif'))

    slide.close()

slide_path = 'P44F_PAS.svs'  
output_dir = 'windows' 
window_size = 512  
stride = 256  # Overlap between windows

save_windows_from_wsi(slide_path, output_dir, window_size, stride)
