import os
openslide_bin_path = r'C:\Users\Jingy\OneDrive\Desktop\nuclei_segmentation\openslide-bin-4.0.0.2-windows-x64\bin'
os.add_dll_directory(openslide_bin_path)
import openslide
from PIL import Image
import os

def patchify_svs(svs_path, output_dir, patch_size=560, stride=560):
    """
    Extracts patches from a .svs file and saves them as images.
    
    Parameters:
    - svs_path: Path to the .svs file.
    - output_dir: Directory to save the patches.
    - patch_size: The size of the square patches.
    - stride: The stride with which the patches are extracted.
    """
    # Open the .svs file
    slide = openslide.OpenSlide(svs_path)
  
    # Calculate the number of patches
    w, h = slide.dimensions
    n_patches_x = (w - patch_size) // stride + 1
    n_patches_y = (h - patch_size) // stride + 1
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Add this after opening the slide to verify that the slide can be read

    # Extract patches
    patch_num = 0
    for x in range(0, n_patches_x * stride, stride):
        for y in range(0, n_patches_y * stride, stride):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert("RGB")  # Convert to RGB
            
            patch_path = os.path.join(output_dir, f'patch_{patch_num}.png')
            patch.save(patch_path)
            print(f'Saved patch {patch_num}: {patch_path}')
            patch_num += 1

    slide.close()

# Example usage
svs_path = 'P44F_PAS.svs'
output_dir = 'patched_images'
patchify_svs(svs_path, output_dir)