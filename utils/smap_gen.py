import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm

import os
from multiprocessing import Pool

def generate_scale_map(file_path):
    try:
        df = pd.read_table(file_path, sep=' ', header=None)
    except:
        split = file_path.split('/')[-3]
        basename = file_path.split('/')[-1].split('.')[0]
        img_path = os.path.join('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu', split, basename + '.jpg')
        smap_path = img_path.replace('.jpg', '_smap.npy')
        if not os.path.exists(smap_path):
            img = Image.open(img_path)
            smap = np.zeros((img.height, img.width))
            np.save(smap_path, smap)
        return

    split = file_path.split('/')[-3]
    basename = file_path.split('/')[-1].split('.')[0]
    img_path = os.path.join('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu', split, basename + '.jpg')
    smap_path = img_path.replace('.jpg', '_smap.npy')

    if os.path.exists(smap_path):
        return

    img = Image.open(img_path)
    smap = np.zeros((img.size[1], img.size[0]))
    if len(df) > 0:
        for _, row in df.iterrows():
            x = row[0]
            y = row[1]
            w = row[2]
            h = row[3]
            scale = np.sqrt(w * h)
            hw = w // 2
            hh = h // 2
            x1 = np.maximum(0, x - hw)
            x2 = np.minimum(img.size[0], x + hw)
            y1 = np.maximum(0, y - hh)
            y2 = np.minimum(img.size[1], y + hh)
            smap[y1:y2, x1:x2] = scale

    np.save(smap_path, smap)

if __name__ == '__main__':
    splits = ['train', 'val', 'test']
    files = []
    for split in splits:
        files += glob(f'/mnt/home/zpengac/USERDIR/Crowd_counting/jhu_crowd_v2.0/{split}/gt/*.txt')

    with Pool(8) as p:
        r = list(tqdm(p.map(generate_scale_map, files, chunksize=8)))
