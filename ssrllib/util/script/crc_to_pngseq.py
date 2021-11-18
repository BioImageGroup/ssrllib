from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from glob import glob
from tqdm import tqdm
import os


if __name__ == "__main__":
    image = '/home/nicola/data/crc/7_UC/54_H_UC_UC_UC_0.svs'
    out_folder = '/home/nicola/data/crc_pngseq'
    size = 512

    class_dir = image.split('/')[-2]
    new_dir = os.path.join(out_folder, class_dir.split('/')[-1])
    os.makedirs(new_dir, exist_ok=True)

    main_slide_name = image.split('/')[-1][:-4]
    slide = OpenSlide(image)
    data_gen = DeepZoomGenerator(slide, tile_size=size, overlap=0, limit_bounds=True)

    max_zoom = data_gen.level_count - 1
    max_tile = data_gen.level_tiles[max_zoom]
    tot_tiles = data_gen.level_count

    # tot = max_tile[0] * max_tile[1]
    tot = 100
    idx = 0


    crop_idx = 0
    for x in range(max_tile[0]):
    
        for y in range(max_tile[1]):
    
            tile = data_gen.get_tile(level=max_zoom, address=(x, y))
            if tile.size == (512, 512):
                tile.save(os.path.join(new_dir, main_slide_name + f'_crop_{crop_idx}_pos_{x:02}-{y:02}.png'))
                crop_idx += 1
                print(f'\r{crop_idx}/{tot}\t', end='')

                if crop_idx >= tot:
                    exit()

            