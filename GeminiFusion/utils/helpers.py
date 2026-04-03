import numpy as np
import torch
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
import cv2
import os

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
logger = None


def print_log(message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


def make_validation_img(img_, depth_, lab, pre):
    cmap = np.load('./utils/cmap.npy')
    
    img = np.array([i * IMG_STD.reshape((3, 1, 1)) + IMG_MEAN.reshape((3, 1, 1)) for i in img_])
    img *= 255
    img = img.astype(np.uint8)
    img = np.concatenate(img, axis=1)

    # HAG may be provided as 1 channel or repeated across 3 channels.
    depth_ = depth_[0].transpose(1, 2, 0)
    if depth_.ndim == 3 and depth_.shape[2] > 1:
        depth_ = depth_[:, :, 0]
    else:
        depth_ = np.squeeze(depth_)
    depth_ = depth_.astype(np.float32)
    depth_ = depth_ / max(float(depth_.max()), 10.0)
    vmax = float(np.percentile(depth_, 95))
    vmin = float(depth_.min())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    depth = (mapper.to_rgba(depth_)[:, :, :3] * 255).astype(np.uint8)
    lab = np.concatenate(lab)
    lab = np.array([cmap[i.astype(np.uint8) + 1] for i in lab])

    pre = np.concatenate(pre)
    pre = np.array([cmap[i.astype(np.uint8) + 1] for i in pre])
    img = img.transpose(1, 2, 0)

    return np.concatenate([img, depth, lab, pre], 1)
