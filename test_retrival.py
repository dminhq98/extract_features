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

def get_files(path):
    file_list, dir_list = [], []
    for dir, subdirs, files in os.walk(path):
        file_list.extend([FJoin(dir, f) for f in files])
        dir_list.extend([FJoin(dir, d) for d in subdirs])
    file_list = filter(lambda x: not os.path.islink(x), file_list)
    dir_list = filter(lambda x: not os.path.islink(x), dir_list)
    return file_list, dir_list

def check_file(path):
    if not os.path.exists('error_image'):
        os.makedirs('error_image')
    for p in path:
        try:
            Image.open(p)
        except IOError:
            path_err = 'error_image'
            shutil.move(p, path_err)
            print("file {} error.".format(p))

def load_features(file_name = 'features.h5'):
  f = h5py.File(file_name, 'r')
  features = f['features'][:]
  print(features.shape)
  path_images = f['path_images']
  path_images = list(path_images)
  print(len(path_images))
  return features,path_images

def test(path,index,model,n=1):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    print(path)
    img = Image.open(path)
    image = transform(img).unsqueeze(0).cuda()
    feature = model(image).data.cpu().numpy().reshape(-1, )

    k = index.get_nns_by_vector(feature, n, include_distances=True)
    return k

def search_knn(path, model, index_file,features_file="features.h5",test_forder="result_test",n=10):
    features, path_images=load_features(file_name = features_file)
    f = len(features[0])
    from annoy import AnnoyIndex
    import random
    u = AnnoyIndex(f, 'angular')
    u.load(index_file)  # super fast, will just mmap the file

    if not os.path.exists(test_forder):
        os.makedirs(test_forder)
    for p in path:
        res = test(p,u,model,10)
        parent, image_names = os.path.split(p)
        image_name = image_names[:-4]
        dect = os.path.join(test_forder,image_name)
        if not os.path.exists(dect):
            os.makedirs(dect)
        destination = os.path.join(dect, image_names)
        shutil.copyfile(p, destination)
        for i,im in enumerate(res[0]):
            name = "Rank {}".format(i+1)+".jpg"
            destination = os.path.join(dect,name)
            shutil.copyfile(path_images[im], destination)

def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--test-forder',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the image folder to test')

    parser.add_argument(
        '--model-name',
        default='resnet50',
        type=str,
        help='Name pretrained model to extract features from ,Default extract feature before layer fc.')

    parser.add_argument(
        '--features-name',
        metavar='PATH',
        type=str,
        required=True,
        help='File features as HDF5 from this location.')

    parser.add_argument(
        '--index-name',
        metavar='PATH',
        type=str,
        required=True,
        help='Index as AnnoyIndex to this location.')

    parser.add_argument(
        '--path-result',
        metavar='PATH',
        type=str,
        default="result_test",
        help='Save result to this location.')

    parser.add_argument(
        '--output_log',
        help='Output file to log to. Default: --output_features + ".log"')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[args.model_name](pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    # model.cuda()
    model.eval()

    path_image = args.test_forder
    images = get_files(path_image)
    images = images[0]

    search_knn(images, model, args.index_name,args.features_name,args.path_result,n=10)
if __name__ == "__main__":

    main()