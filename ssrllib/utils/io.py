import datetime
from glob import glob
import os
from typing import List
import xml.etree.ElementTree as et
from PIL import Image
import numpy as np

from openslide import OpenSlide
import logging


def print_ts(text: str):
    """Prints text to stdout and includes timestamps at the beginning of each line

    Args:
        text ([type]): Text to be incapsulated in the datetime wrapper
    """
    print('[%s] %s' % (datetime.datetime.now(), text), flush=True)


def load_images(images: List[str], type: str) -> List:
    """Open a list of images files from filesystem

    Args:
        images (List[str]): List of paths for the image files
        type (str): Files extension 

    Returns:
        List: List of memory-loaded images
    """
    slides = []

    for image in images:
        if type == 'ndpi':
            slide = OpenSlide(image)
        elif type == 'svs':
            slide = OpenSlide(image)
        elif type == 'png':
            slide = np.array(Image.open(image))
        else:
            NotImplementedError(f'The required format {type} is not implemented yet')

        slides.append(slide)

    return slides


def load_rois(xmls: List[str]) -> List:
    """Open a list of xml files and extract the coordinates for the ROIs

    Args:
        xmls (List[str]): List of paths for the xml files

    Returns:
        List: List of parsed xml files
    """
    rois = []
    for xml in xmls:
        tree = et.parse(xml)
        root = tree.getroot()
        rois.append(root[0])

    return rois


def get_class_files(data_dir: str, type: str = 'ndpi') -> List[str]:
    """Gathers all files of a given type from the filesystem, for each available class. Files are sorted alphanumerially
    to increase robustness.

    Args:
        data_dir (str): Root directory to search for files in
        type (str, optional): Extention of files to be collected. Defaults to 'ndpi'.

    Returns:
        List[str]: List of paths to the files
    """

    files = []
    for class_dir in sorted(glob(os.path.join(data_dir, f"*"))):
        files.extend(sorted(glob(os.path.join(class_dir, f"*.{type}"))))

    return files


def get_class_labels(data_dir, type='ndpi') -> List[int]:
    """Gathers all class labels from folder structure

    Args:
        data_dir (str): Root directory to search for files in
        type (str, optional): Extention of files to be collected. Defaults to 'ndpi'.

    Returns:
        List[int]: [description]
    """

    classes = []

    for class_dir in sorted(glob(os.path.join(data_dir, f"*"))):
        class_idx = int(class_dir.split('/')[-1].split('_')[0])
        classes.extend([class_idx] * len(glob(os.path.join(class_dir, f"*.{type}"))))

    return np.array(classes)
