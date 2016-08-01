"""
Paths and other variables for different datasets. See the ARIDataset for
for an example hoe to write your own dataset definition
"""
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import skimage.io


class ARIDataset:
    def __init__(self):

        self.im_dir = '/zfsauton/home/aladdha/seman_exps/ARI/data/images'
        self.gt_dir = '/zfsauton/home/aladdha/seman_exps/ARI/data/gt'

        self.data_dir = '/zfsauton/home/aladdha/seman_exps/ARI/data/'

        self.list_dir = '/zfsauton/home/aladdha/seman_exps/ARI/data/lists'

        self.nlabels = 13
        self.ignore_label = 255  # Don't care label

        self.label_names = ['sky',
                            'water',
                            'asphalt_h',
                            'concrete_h',
                            'other_h',
                            'brick_v',
                            'concrete_v',
                            'wood_v',
                            'vehicle',
                            'tree',
                            'people',
                            'metal_o',
                            'other_o']
        self.colormap = np.array([[0, 0, 255], [255, 0, 0], [128, 128, 128],
                                 [128, 0, 128], [0, 128, 0], [0, 255, 0],
                                 [255, 128, 0], [255, 255, 0], [255, 0, 255],
                                 [255, 50, 128], [255, 0, 128], [128, 50, 0],
                                 [0, 255, 255]], np.float32)
        self.image_ext = 'jpg'

    def get_gt(self, fname):
        labName = os.path.join(self.gt_dir, '{}.txt'.format(fname))
        return np.loadtxt(labName)

    def get_im(self, fname):
        im_name = os.path.join(self.im_dir,
                    '{}.{}'.format(fname, self.image_ext))
        return skimage.io.imread(im_name)

    def get_list_path(self, list_name):
        return os.path.join(self.list_dir, '{}.txt'.format(list_name))


datasets = {}
datasets['ARI'] = ARIDataset()