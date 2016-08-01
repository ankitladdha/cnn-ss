"""
This file contains the utillities required to read and write files - text and binary
"""
from __future__ import division

__author__ = "Ankit Laddha <aladdha@andrew.cmu.edu>"


def read_list_file(listFileName):
    """
    Load a list of file names

    Parameters
    ----------
    listFileName : string
            Path of the file containing the list

    Returns
    -------
    listFiles : list of strings
            A list of all the file names
    """
    listFilePtr = open(listFileName, 'r')
    listFiles = [line.strip() for line in listFilePtr]
    return listFiles


def write_list_file(listFileName, fileList):
    """
    Write a list of file names

    Parameters
    ----------
    listFiles : list of strings
            A list of all the file names
    listFileName : string
            Path of the file containing the list
    """
    listFilePtr = open(listFileName, 'w')
    for line in fileList:
        listFilePtr.write('{}\n'.format(line))
    listFilePtr.close()
