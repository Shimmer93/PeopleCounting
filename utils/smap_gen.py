import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from scipy.spatial import KDTree

import os
from multiprocessing import Pool

def get_ratio(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
        else:
            ratio = 1.0
    return ratio

def generate_smap(img, orig_img, pts, df):
    if len(pts) == 0 or pts is None:
        return np.zeros((img.size[1], img.size[0]))

    ratio = get_ratio(orig_img.height, orig_img.width, 512, 2048)

    temp_smap = np.zeros((img.size[1], img.size[0]))
    if len(df) > 0:
        for _, row in df.iterrows():
            x = int(row[0] * ratio)
            y = int(row[1] * ratio)
            w = int(row[2] * ratio)
            h = int(row[3] * ratio)
            scale = np.sqrt(w * h)
            hw = w // 2
            hh = h // 2
            x1 = np.maximum(0, x - hw)
            x2 = np.minimum(img.size[0], x + hw)
            y1 = np.maximum(0, y - hh)
            y2 = np.minimum(img.size[1], y + hh)
            temp_smap[y1:y2, x1:x2] = scale

    smap = temp_smap.copy()

    pts = pts[pts[:, 0] < smap.shape[1]]
    pts = pts[pts[:, 1] < smap.shape[0]]

    leafsize = 2048
    tree = KDTree(pts.copy(), leafsize=leafsize)

    for i in range(smap.shape[0]):
        for j in range(smap.shape[1]):
            if smap[i, j] > 0:
                continue
            _, idx = tree.query([j, i], k=1)
            smap[i, j] = temp_smap[int(pts[idx, 1]), int(pts[idx, 0])]

    return smap

def run(img_path):
    split = img_path.split('/')[-2]
    basename = img_path.split('/')[-1].split('.')[0]
    orig_img_path = os.path.join('/mnt/home/zpengac/USERDIR/Crowd_counting/jhu_crowd_v2.0', split, 'images', basename + '.jpg')
    pts_path = img_path.replace('.jpg', '.npy')
    df_path = os.path.join('/mnt/home/zpengac/USERDIR/Crowd_counting/jhu_crowd_v2.0', split, 'gt', basename + '.txt')
    smap_path = img_path.replace('.jpg', '_smap.png')

    img = Image.open(img_path)
    orig_img = Image.open(orig_img_path)
    pts = np.load(pts_path)
    try:
        df = pd.read_table(df_path, sep=' ', header=None)
    except:
        df = None

    smap = generate_smap(img, orig_img, pts, df)

    smap_img = Image.fromarray(smap.astype(np.uint8))
    smap_img.save(smap_path)

if __name__ == '__main__':
    img_fns = glob('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu/*/*.jpg')

    with Pool(8) as p:
        r = list(tqdm(p.map(run, img_fns), total=len(img_fns)))
