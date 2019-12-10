import importlib
from tqdm import tqdm
from torchvision.datasets.folder import default_loader
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
import h5py
import shutil
from os import path
import sys
import random
import logging
import argparse
import os
from PIL import Image

FJoin = os.path.join


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


def load_model(model, model_file):
    checkpoint = torch.load(model_file)
    # Support for checkpoints saved by scripts based off of
    #   https://github.com/pytorch/examples/blob/master/imagenet/main.py
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    logging.info('Loading model from %s', model_file)
    model.load_state_dict(checkpoint, strict=False)

    missing_keys = set(model.state_dict().keys()) - set(checkpoint.keys())
    extra_keys = set(checkpoint.keys()) - set(model.state_dict().keys())
    if missing_keys:
        logging.info('Missing keys in --model-file: %s.', missing_keys)
    if extra_keys:
        logging.info('Extra keys ignored in --model-file: %s.', extra_keys)


class ListDataset(torch.utils.data.Dataset):
    def __init__(self,
                 images_list,
                 transform=None,
                 loader=default_loader):
        self.images_list = images_list
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.images_list[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.images_list)


def image_path_to_name(image_path):
    # return np.string_(path.splitext(path.basename(image_path))[0])
    parent, image_name = path.split(image_path)
    image_name = path.splitext(image_name)[0]
    parent = path.split(parent)[1]
    return path.join(parent, image_name)


def extract_features_to_disk(image_paths,
                             model,
                             batch_size,
                             workers,
                             output_hdf5):
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = ListDataset(image_paths,
                          transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              normalize,
                          ]))
    # if torch.cuda.is_available():
    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=workers,
    #         pin_memory=True)
    # else:
    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=0,
    #         pin_memory=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True)
    features = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (input_data, paths) in enumerate(tqdm(loader)):
        input_var = torch.autograd.Variable(
            input_data, volatile=True).to(device)
        current_features = model(input_var).data.cpu().numpy()
        for j, image_path in enumerate(paths):
            features[image_path] = current_features[j].reshape(-1, )

    feature_shape = features[list(features.keys())[0]].shape
    logging.info('Feature shape: %s' % (feature_shape,))
    logging.info('Outputting features')

    if sys.version_info >= (3, 0):
        string_type = h5py.special_dtype(vlen=str)
    else:
        string_type = h5py.special_dtype(vlen=unicode)  # noqa
    paths = features.keys()
    logging.info('Stacking features')
    features_stacked = np.vstack([features[path] for path in paths])
    logging.info('Output feature size: %s' % (features_stacked.shape,))
    with h5py.File(output_hdf5, 'a') as f:
        f.create_dataset('features', data=features_stacked)
        f.create_dataset(
            'path_images',
            (len(paths),),
            dtype=string_type)
        # For some reason, assigning the list directly causes an error, so we
        # assign it in a loop.
        for i, image_path in enumerate(paths):
            # f['image_names'][i] = image_path_to_name(image_path)
            f['path_images'][i] = image_path


def _set_logging(logging_filepath):
    """Setup logger to log to file and stdout."""
    log_format = '%(asctime)s.%(msecs).03d: %(message)s'
    date_format = '%H:%M:%S'

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    logging.info('Writing log file to %s', logging_filepath)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--image-forder',
        metavar='PATH',
        type=str,
        required=True,
        help='path to the image folder to extract')
    parser.add_argument(
        '--model-name',
        default='resnet50',
        type=str,
        help='Name pretrained model to extract features from ,Default extract feature before layer fc.')

    parser.add_argument(
        '--output-features',
        metavar='PATH',
        type=str,
        required=True,
        help='Output features as HDF5 to this location.')
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers')
    parser.add_argument(
        '--batch-size',
        default=256,
        type=int,
        metavar='N')

    parser.add_argument(
        '--output_log',
        help='Output file to log to. Default: --output_features + ".log"')

    args = parser.parse_args()

    # assert not path.exists(args.output_features)
    if args.output_log is None:
        args.output_log = args.output_features + '.log'
    _set_logging(args.output_log)
    logging.info('Args: %s', args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.__dict__[args.model_name](pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    # model.cuda()
    model.eval()
    # print (model)
    path_image = args.image_forder
    images = get_files(path_image)
    images = list(images[0])
    # print(images)
    print("Checking image...")
    check_file(images)
    print("Checking imageed")
    images = get_files(path_image)
    images = list(images[0])
    extract_features_to_disk(images, model, args.batch_size,
                             args.workers, args.output_features)
    # print(images)
    # print(args.model_name)
    # print(args.workers)
    # print(args.batch_size)
    # print(args.output_features)


if __name__ == "__main__":
    main()
