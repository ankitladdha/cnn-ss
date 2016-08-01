"""
This file contains utilities used for evaluation
"""
from __future__ import division

__author__ = "Ankit Laddha <aladdha@andrew.cmu.edu>"

import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_confusion_matrix(gt, pred, noCareLabel=255,
                               nLabels=None, labels2use=None):
    """
    Calculate the confusion matrix between the ground truth vector
    and predicted vector

    Parameters
    ----------
    gt : numpy array (nx1)
        An array of gt labels
    pred : numpy array (nx1)
        An array of predicted labels
    noCareLabel : number
        Don't care label. The pixel with this round truth
        will be ignored during the calculations
    nLabels : number
        Number of labels to be considered
    labels2use : numpy array
        A list of labels to be used to calculate the confusion matrix

    Returns
    -------
    confmat : numpy array
        Calculated confusion_matrix
    """

    if labels2use is None:
        labels2use = np.arange(nLabels)

    if noCareLabel is not None:
        mask = gt != noCareLabel
    else:
        mask = (np.ones(gt.shape) == 1)

    confmat = confusion_matrix(gt[mask], pred[mask], labels2use)
    return confmat


def get_pixel_stats(confmat):
    """
    Get the true positives, false positives and false negatives for each
    of the labels using the confusion matrix

    """
    tp = np.diag(confmat)
    fp = (confmat.sum(0)).T - tp
    fn = confmat.sum(1) - tp

    return tp, fp, fn


def get_various_accu(confmat):
    """
    Get various accuracy measures using a confusion matrix

    Parameters
    ----------
    confmat : numpy array
        Confusion matrix to calculate the accuracies

    Returns
    -------
    () : a tuple of various acuracy measures
        (Per Class Accuracy, Per Class Recall, Per Class F1, Per Class IOU,
        Mean Accuracy, Mean IOU, Mean F1,
        Per Pixel Accuracy, Mask of classes present in the ground truth)
    """
    (tp, fp, fn) = get_pixel_stats(confmat)

    # Find if a class is present in the ground truth or not.
    # If a classes is not present then we ignore it
    mask = tp+fn > 0
    tp = tp[mask]
    fp = fp[mask]
    fn = fn[mask]

    precision = tp/(tp+fp+1e-10)
    recall = tp/(tp+fn+1e-10)
    f = 2*(precision*recall)/(precision+recall)

    jaccard = tp/(tp+fp+fn+1e-10)
    perPixel = sum(tp)/confmat.sum()

    return (precision*100, recall*100, f*100, jaccard*100,
            precision.mean()*100, jaccard.mean()*100, f.mean()*100,
            perPixel*100, mask)


def write_results(fileName, confMat, labels):
    """
    Write all the results in a file

    Parameters
    ----------
    fileName : string
        Path of the file for the results to be written
    confmat : numpy array
        Confusion matrix to calculate the accuracies
    labels : list of string
        Names of all the Labels present in the datatset

    Returns
    -------
    () : a tuple of various acuracy measures
        (Per Class Accuracy, Per Class Recall, Per Class F1, Per Class IOU,
        Mean Accuracy, Mean IOU, Mean F1,
        Per Pixel Accuracy)
    """
    fid = open(fileName, 'w')
    (precision, recall, F1, jaccard, perClass,
    avgJaccard, avgF1, perPixel, mask) = get_various_accu(confMat)
    actual_lab_no = 0
    for labNo, label in enumerate(labels):
        fid.write('{}:\n'.format(label))
        if mask[labNo]:
            fid.write('Precision: {}\n'.format(precision[actual_lab_no]))
            fid.write('Recall: {}\n'.format(recall[actual_lab_no]))
            fid.write('F1: {}\n'.format(F1[actual_lab_no]))
            fid.write('IOU: {}\n'.format(jaccard[actual_lab_no]))
            actual_lab_no += 1
        else:
            fid.write('Not Present in GT')
        fid.write('\n\n')

    fid.write('All Classes Accuracies:\n')
    fid.write('Per Class Accuracy: {}\n'.format(perClass))
    fid.write('Average Jaccard: {}\n'.format(avgJaccard))
    fid.write('Average F1: {}\n'.format(avgF1))
    fid.write('Per Pixel Accuracy: {}\n'.format(perPixel))
    fid.close()
    return (confMat, precision, recall, F1, jaccard,
            perClass, avgJaccard, avgF1, perPixel)
