import argparse
import os
import random

from glob import glob

def main(path, output_path, num_splits):
    fns = glob(os.path.join(path, '*.jpg')) + \
          glob(os.path.join(path, '*.jpeg')) + \
          glob(os.path.join(path, '*.png'))
    num_fs = len(fns)
    random.shuffle(fns)
    fns = [fn+'\n' for fn in fns]

    if num_splits == 2:
        splits = ['train', 'val']
        fn_groups = [fns[:int(0.8*num_fs)], fns[int(0.8*num_fs):]]
    elif num_splits == 3:
        splits = ['train', 'val', 'test']
        fn_groups = [fns[:int(0.8*num_fs)], fns[int(0.8*num_fs):int(0.9*num_fs)], fns[int(0.9*num_fs):]]
    else:
        raise ValueError('Only support 2 or 3 splits')

    os.makedirs(output_path, exist_ok=True)
    for split, fn_group in zip(splits, fn_groups):
        with open(os.path.join(output_path, f'{split}.txt'), 'w') as f:
            f.writelines(fn_group)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--num_splits', type=int, default=3)
    args = parser.parse_args()

    main(args.path, args.output_path, args.num_splits)