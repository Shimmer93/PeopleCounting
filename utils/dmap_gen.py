import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
 
import os
import argparse

def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 3:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = 15 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    path = args.path
    output_path = args.output_path
    dataset = args.dataset

    if dataset == 'SmartCity':
        images = path + '/images'
        anno = path + '/images'
        gt_colname = 'loc'
        gt_suffix = ''
    elif dataset == 'UCF_CC_50':
        images = path + ''
        anno = path + ''
        gt_colname = 'annPoints'
        gt_suffix = '_ann'

    os.makedirs(output_path, exist_ok=True)

    for fn in tqdm(os.listdir(images)):
        if fn.endswith('.jpg'):
            img = cv2.imread(os.path.join(images, fn))
            fn_gt = os.path.join(anno, fn.split('.')[0]+f'{gt_suffix}.mat')
            label = loadmat(fn_gt)
            annPoints = label[gt_colname]
            im_density = gaussian_filter_density(img, annPoints)
            np.save(os.path.join(output_path, fn.split('.')[0]+'.npy'), im_density)
