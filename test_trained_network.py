"""
Test the network and store results
"""

from __future__ import division
from __future__ import print_function

import caffe

import numpy as np
from skimage.io import imsave
import scipy.misc
import scipy.ndimage
import os
from tqdm import tqdm
import argparse

from tools import SimpleTransformer

from util_extra import check_and_make_dir
from util_files import read_list_file
import utils_metrics as um
import util_images as ui
import datasets as dataset_utils
import IPython as ip


class Network(object):
    """
    Class to encapsulate the network running part
    """

    def __init__(self, deploy_fname, weights_fname, transformer, pred_layer):
        self.net = net = caffe.Net(deploy_fname, weights_fname, caffe.TEST)
        self.transformer = transformer
        self.pred_layer = pred_layer

    def get_predictions(self, im, isargmax=True):
        im = self.transformer.preprocess(im)

        self.net.blobs['data'].reshape(1, 3, im.shape[1], im.shape[2])
        self.net.blobs['data'].data[0, :] = im
        output = self.net.forward()

        pred = output[self.pred_layer]
        pred = np.squeeze(pred)

        if pred.shape[1] != im.shape[1] or pred.shape[2] != im.shape[2]:
            resize_pred = np.zeros((im.shape[1], im.shape[2], dataset.nlabels))
            for cls in range(dataset.nlabels):
                resize_pred[:, :, cls] = scipy.ndimage.zoom(pred[cls, :, :], (im.shape[1] / pred.shape[1],
                                                                          im.shape[2] / pred.shape[2]), order=1)
        else:
            resize_pred = pred

        if isargmax:
            preds = resize_pred.argmax(axis=2)
        else:
            preds = resize_pred

        return preds


def test_network(all_fnames, net, dataset, results_fname,
                 do_overlay=False, overlay_dir=None, colormap=None, alpha=0.5):

    total_confmat = np.zeros((dataset.nlabels, dataset.nlabels))
    for (iter, fname) in tqdm(enumerate(all_fnames)):
        im = dataset.get_im(fname)
        gt = dataset.get_gt(fname)

        pred = net.get_predictions(im)

        if do_overlay:
            overlay_im_fname = os.path.join(overlay_dir, '{}.png'.format(fname))
            overlayed_im = ui.overlay_labels(pred, im, colormap, alpha)
            imsave(overlay_im_fname, overlayed_im)

        confmat = um.calculate_confusion_matrix(gt, pred,
                   noCareLabel=dataset.ignore_label, nLabels=dataset.nlabels)
        total_confmat += confmat

    results = um.write_results(results_fname, total_confmat, dataset.label_names)
    return results


def build_parser():
    """
    Build the argument parser for the program

    :return:
     parser: an instance of the argparse class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=-1, type=int,
                        help='Which GPU to use? If want to use CPU then use -1')
    parser.add_argument('--exp-name', default='ari_fold_2',
                        help='Name of the Experiment')
    parser.add_argument('--dataset', default='ARI',
                        help='Dataset to use')
    parser.add_argument('--test-list', default='fold_test_2_20',
                        help='List of files names to test')
    parser.add_argument('--pred-layer', default='fc8_full_ss',
                        help='Layer name to be used as prediction layer')
    parser.add_argument('--do-overlay', default=True, type=bool,
                help='Overlay the predictions on the images and save or not')
    parser.add_argument('--iters', metavar='N', type=int, nargs='+',
                help='Network iterations for which we want predictions')
    return parser


if __name__ == '__main__':

    # Parse the arguments
    parser = build_parser()
    args = parser.parse_args()

    gpu = args.gpu
    if gpu < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)

    exp_name = args.exp_name
    dataset_name = args.dataset
    listName = args.test_list
    pred_layer = args.pred_layer
    do_overlay = args.do_overlay
    network_iters = args.iters


    dataset = dataset_utils.datasets[dataset_name]

    # Load the image file names
    list_filename = os.path.join(dataset.list_dir, '{}.txt'.format(listName))
    all_fnames = read_list_file(list_filename)

    # Data
    data_dir = dataset.data_dir

    # Deploy File
    deploy_fname = os.path.join(data_dir, 'cnn', 'configs', 'deploy.prototxt')

    # Directory to store results
    results_dir = os.path.join(data_dir, 'cnn', 'results', exp_name)
    check_and_make_dir(results_dir)

    weights_dir = os.path.join(data_dir, 'cnn', 'models', exp_name)

    # Setup a Transformer for image
    bgr_mean = np.array([104, 117, 123])
    transformer = SimpleTransformer(bgr_mean)

    colormap = dataset.colormap
    colormap = colormap.tolist()
    alpha = 0.65

    all_network_iters = np.array(network_iters)
    all_avg_jaccard = np.zeros(all_network_iters.size)
    for i, network_iter in enumerate(network_iters):
        weights_fname = os.path.join(weights_dir,
                         'train__iter_{}.caffemodel'
                                     .format(network_iter))
        txt_results_fname = os.path.join(results_dir, '{}_{}.txt'
                                         .format(listName, network_iter))
        npy_results_fname = os.path.join(results_dir, '{}_{}.npy'
                                         .format(listName, network_iter))
        overlay_dir = os.path.join(results_dir, 'vis_{}'.format(network_iter))
        check_and_make_dir(overlay_dir)

        if not os.path.isfile(npy_results_fname):
            if not os.path.isfile(weights_fname):
                print('Weights file not present : {}'.format(weights_fname))
                continue
            else:
                network = Network(deploy_fname, weights_fname, transformer, pred_layer)
                results = test_network(all_fnames, network, dataset, txt_results_fname,
                                       do_overlay, overlay_dir, colormap, alpha)
                np.save(npy_results_fname, results)
        else:
            results = np.load(npy_results_fname)
        all_avg_jaccard[i] = results[5]


    ip.embed()
