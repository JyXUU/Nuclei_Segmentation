import openslide
import numpy as np
import matplotlib.pyplot as plt
import openslide
import tifffile as tiff
slide_path = 'P44F_PAS.svs'
slide = openslide.OpenSlide(slide_path)
from openslide.deepzoom import DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
print("The number of levels in the tiles object are: ", tiles.level_count)
print("The dimensions of data in each level are: ", tiles.level_dimensions)
#Total number of tiles in the tiles object
print("Total number of tiles = : ", tiles.tile_count)

import os
###### Saving each tile to local directory
orig_tile_dir_name = 'Patches'
cols, rows = tiles.level_tiles[15]
i=1
for row in range(rows):
    for col in range(cols):
        tile_name =   'patch_' + str(i)      #str(col) + "_" + str(row)
        #tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        #print("Now processing tile with title: ", tile_name)
        temp_tile = tiles.get_tile(15, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        print(temp_tile_np.mean(), temp_tile_np.std(), os.path.join(orig_tile_dir_name, tile_name) + "_original.tif")
        #Save original tile
        if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
            i+=1
            tiff.imwrite(os.path.join(orig_tile_dir_name, tile_name) + "_original.tif", temp_tile_np)
#         if temp_tile_np.mean() < 230 and temp_tile_np.std() > 15:
#             print("Processing tile number:", tile_name)
#             norm_img, H_img, E_img = norm_HnE(temp_tile_np, Io=240, alpha=1, beta=0.15)
#         #Save the norm tile, H and E tiles
            
#             tiff.imsave(norm_tile_dir_name+tile_name + "_norm.tif", norm_img)
#             tiff.imsave(H_tile_dir_name+tile_name + "_H.tif", H_img)
#             tiff.imsave(E_tile_dir_name+tile_name + "_E.tif", E_img)
            
        else:
            # print("NOT PROCESSING TILE:", tile_name)
            pass
