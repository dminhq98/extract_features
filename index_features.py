import os
from PIL import Image
FJoin = os.path.join
import argparse
import logging
import random
import sys
from os import path
import shutil

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torchvision.datasets.folder import default_loader
from tqdm import tqdm




def load_features(file_name = 'features.h5'):
  f = h5py.File(file_name, 'r')
  features = f['features'][:]
  print(features.shape)
  path_images = f['path_images']
  path_images = list(path_images)
  print(len(path_images))
  return features,path_images

def index_knn(features,file_name ):

    from annoy import AnnoyIndex
    import random
    f = len(features[0])
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
    for i in range(len(features)):
        v = features[i]
        t.add_item(i, v)

    t.build(10) # 10 trees
    t.save(file_name)

  # u = AnnoyIndex(f, 'angular')
  # u.load('feature_space.ann') # super fast, will just mmap the file
  # return u

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--features-name',
        metavar='PATH',
        type=str,
        required=True,
        help='File features as HDF5 from this location.')

    parser.add_argument(
        '--output-index',
        metavar='PATH',
        type=str,
        required=True,
        help='Output index as AnnoyIndex to this location.')

    parser.add_argument(
        '--output_log',
        help='Output file to log to. Default: --output_features + ".log"')

    args = parser.parse_args()

    print(args.features_name)
    print(args.output_index)

    features, path_images = load_features(args.features_name)
    index_knn(features, args.output_index)
if __name__ == "__main__":

    main()
