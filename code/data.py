import argparse
import functools
import glob
import os
import shutil
import sys
import tarfile
from zipfile import ZipFile
import random

from PIL import Image
import ipdb
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import center_crop
from tqdm import tqdm, trange

_DEFAULT_DATA_ROOT = 'data'


# Can't use lambda func due to distributed training
def _convert(x):
    return x.convert('RGB')

def _tobyte(x):
    return (x * 255).byte()

def _maxcentercrop(x):
    min_dim = min([x.shape[1], x.shape[2]])
    return center_crop(x, (min_dim, min_dim))



def strided_batch1d(data, *, batch_size, window_size, stride=None, drop_last=False):
    assert data.ndim == 1
    B, X, T = batch_size, len(data), window_size
    S = (stride or T//2)
    offset = (X - T) % S
    toks_per_batch = T + S * (B - 1)
    n_strides = (X - offset - T) // S + 1
    n_full_batches = n_strides // B

    # Rest of the data
    for i in trange(n_full_batches):
        # Get strided batch
        x_flat = data[i*B*S : i*B*S+toks_per_batch]
        assert x_flat.size(-1) == toks_per_batch
        x = torch.empty(B, T, dtype=x_flat.dtype, device=x_flat.device)
        for j in range(len(x)):
            x[j,:] = x_flat[j*S:j*S+T]
        mask = torch.zeros_like(x).bool()
        mask[:,-S:] = True

        # Special case: first chunk
        if i == 0:
            mask[0] = True

        yield x, mask

    # Special case: last chunk
    if not drop_last and n_full_batches == 0 or len(data) - (n_full_batches*B*S+T-S) > 0:
        if offset > 0:
            x_flat = data[n_full_batches*B*S : -offset]
        else:
            x_flat = data[n_full_batches*B*S:]
        assert (len(x_flat)-T) % S == 0 and (len(x_flat)-T)//S+1 < B
        x = torch.empty((len(x_flat)-T)//S+1 + int(offset > 0), T, dtype=x_flat.dtype, device=x_flat.device)
        for j in range((len(x_flat)-T)//S+1):
            x[j,:] = x_flat[j*S:j*S+T]
        mask = torch.zeros_like(x).bool()
        mask[:,-S:] = True
        if offset > 0:
            x[-1,:] = data[-T:]
            mask[-1,:-offset] = False
        if n_full_batches == 0:
            mask[0] = True
        yield x, mask


def _prepare_mnist(out_dir):
    temp = os.path.join(out_dir, 'mnist_temp')

    train = datasets.MNIST(root=temp, download=True, train=True)
    test = datasets.MNIST(root=temp, download=True, train=False)

    train_data = train.data.numpy()[..., np.newaxis]
    train_label = train.targets.numpy()
    test_data = test.data.numpy()[..., np.newaxis]
    test_label = test.targets.numpy()

    assert train_data.shape == (60000, 28, 28, 1) and test_data.shape == (10000, 28, 28, 1)
    assert train_data.dtype == test_data.dtype == np.uint8
    assert train_label.shape == (60000,) and test_label.shape == (10000,)
    assert train_label.dtype == test_label.dtype == np.int64

    np.save(os.path.join(out_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(out_dir, 'train_label.npy'), train_label)
    np.save(os.path.join(out_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(out_dir, 'test_label.npy'), test_label)
    import ipdb; ipdb.set_trace()

    shutil.rmtree(temp)
    print('MNIST dataset successfully created.')


def _prepare_cifar10(out_dir):
    temp = os.path.join(out_dir, 'cifar10_temp')

    train = datasets.CIFAR10(root=temp, download=True, train=True).data
    test = datasets.CIFAR10(root=temp, download=True, train=False).data

    assert train.shape == (50000,32,32,3) and test.shape == (10000,32,32,3)
    assert train.dtype == test.dtype == np.uint8

    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)

    shutil.rmtree(temp)
    print('CIFAR10 dataset successfully created.')


def _prepare_oord_imagenet(out_dir, *, size, data_root):
    # Assume the user has extracted the archives into the following folders:
    #     DATA_ROOT/oord_imagenet32/{train,valid}_32x32
    #     DATA_ROOT/oord_imagenet64/{train,valid}_64x64
    # inside the folder f'{DATA_ROOT}/oord_imagenet'

    assert size in (32, 64)

    train_dir = os.path.join(out_dir, f'train_{size}x{size}')
    valid_dir = os.path.join(out_dir, f'valid_{size}x{size}')
    assert os.path.isdir(train_dir) and os.path.isdir(valid_dir)

    train_files = [os.path.join(train_dir, fn) for fn in os.listdir(train_dir)]
    valid_files = [os.path.join(valid_dir, fn) for fn in os.listdir(valid_dir)]
    assert len(train_files) == 1281149 and len(valid_files) == 49999

    def _process(split, files):
        array_fn = os.path.join(out_dir, f'{split}.npy')
        if os.path.exists(array_fn):
            print(f'{array_fn} already exists, skipping.')
        else:
            images = []
            for img_fn in tqdm(files, desc=f'Creating {split} set for OordImageNet {size}x{size}'):
                images.append(np.array(Image.open(img_fn)))
            images = np.stack(images, axis=0)
            np.save(array_fn, images)
            assert images.shape == (len(files), size, size, 3)

    _process('valid', valid_files)
    _process('train', train_files)

    print(f'Downsampled ImageNet {size}x{size} dataset successfully created.')


def _download_url(url, fpath):
    import urllib
    assert not os.path.exists(fpath)
    urllib.request.urlretrieve(url, fpath)


def _prepare_text8(out_dir):
    zip_path = os.path.join(out_dir, 'text8.zip')
    txt_path = os.path.join(out_dir, 'text8')

    assert not os.path.exists(txt_path) and not os.path.exists(zip_path)
    print(f'Downloading text8 zip archive...')
    TEXT8_URL = 'http://mattmahoney.net/dc/text8.zip'
    _download_url(TEXT8_URL, zip_path)

    with ZipFile(zip_path, 'r') as f:
        f.extractall(out_dir)
    assert os.path.exists(zip_path)

    print('Preparing text8... ', end='', flush=True)
    with open(os.path.expanduser(txt_path), 'r') as f:
        all_data = f.read()
    all_data = np.frombuffer(all_data.encode(), dtype='uint8')
    vocab, all_data = np.unique(all_data, return_inverse=True)
    all_data = all_data.astype('uint8')

    assert all_data.shape == (100000000,)
    assert ''.join(map(chr, vocab)) == ' abcdefghijklmnopqrstuvwxyz'

    np.save(os.path.join(out_dir, 'text8.npy'), all_data)
    np.save(os.path.join(out_dir, 'vocab.npy'), vocab)

    os.remove(zip_path)
    os.remove(txt_path)
    print('Done!', flush=True)


def _prepare_celebahq(out_dir, *, size):
    if os.path.exists(os.path.join(out_dir, f'train_{size}x{size}.npy')):
        print(f'CelebA-HQ train_{size}x{size} already exists!')
        return

    import tensorflow as tf
    tar_path = os.path.join(out_dir, 'celeba-tfr.tar')
    tfr_path = os.path.join(out_dir, 'celeba-tfr')
    if not os.path.exists(tfr_path):
        if not os.path.exists(tar_path):
            print(f'Downloading CelebA-HQ tar archive...')
            TFR_URL = 'https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar'
            _download_url(TFR_URL, tar_path)
        assert os.path.exists(tar_path)

        tar_file = tarfile.open(tar_path)
        tar_file.extractall(out_dir)
    assert os.path.isdir(tfr_path) and len(os.listdir(tfr_path)) == 2

    def _resize_image_array(images, size, interpolation=Image.BILINEAR):
        assert type(size) == tuple
        N, origH, origW, C = images.shape  # Assume NHWC
        assert C in (1, 3) and images.dtype == np.uint8

        if size == (origH, origW):
            return images

        resized = []
        for img in images:
            pil = Image.fromarray(img.astype('uint8'), 'RGB')
            pil = pil.resize(size, resample=interpolation)
            resized.append(np.array(pil))

        resized = np.stack(resized, axis=0)
        assert resized.shape == (N, *size, C)

        return resized

    def _process_folder(split):
        split_str = {'train': 'train', 'val': 'validation'}[split]
        filenames = glob.glob(os.path.join(tfr_path, split_str, '*.tfrecords'))
        dataset = tf.data.TFRecordDataset(filenames=filenames)
        processed = []

        for example in tqdm(dataset, desc=f'Processing {split} set...'):
            parsed = tf.train.Example.FromString(example.numpy())
            shape = parsed.features.feature['shape'].int64_list.value
            img_bytes = parsed.features.feature['data'].bytes_list.value[0]
            img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(shape)
            processed.append(img)

        processed = np.stack(processed)
        assert len(processed) == {'train': 27000, 'val': 3000}[split]
        assert processed.shape[1:] == (256, 256, 3)

        out_path = os.path.join(out_dir, f'{split}_{size}x{size}.npy')
        resized = _resize_image_array(processed, (size, size))
        assert resized.shape == (processed.shape[0], size, size, processed.shape[3])
        np.save(out_path, resized)
        print(f'Saved {out_path}')

    _process_folder('train')
    _process_folder('val')
    print(f'CelebA-HQ {size}x{size} successfully created.')


# Old KittiStereo
def _prepare_old_kitti_stereo(out_dir):
    if os.path.exists(os.path.join(out_dir, f'train.npy')):
        print(f'KITTI Stereo train.npy already exists!')
        return

    # TODO: Fix hardcoded path
    KITTI_RAW_PATH = '/data/datasets/kitti'
    SHAPE = (3, 320, 1224)

    def _process_files(split, fn):
        with open(os.path.join(KITTI_RAW_PATH, fn), 'r') as f:
            image_paths = [os.path.join(KITTI_RAW_PATH, line.strip())
                           for line in f.readlines()]
        assert len(image_paths) % 2 == 0

        trfms = transforms.Compose([
            transforms.Lambda(_convert),
            transforms.CenterCrop(SHAPE[1:]),
            transforms.ToTensor(),
            transforms.Lambda(_tobyte),
        ])
        
        images = []
        image_paths = list(zip(image_paths[0::2], image_paths[1::2]))
        for img_left, img_right in tqdm(image_paths):
            img_left = trfms(Image.open(img_left))
            img_right = trfms(Image.open(img_right))
            assert img_left.shape == SHAPE and img_left.dtype == torch.uint8
            assert img_right.shape == SHAPE and img_right.dtype == torch.uint8
            images.append(torch.stack((img_left, img_right), dim=0))
            assert images[-1].shape == (2, *SHAPE)

        images = torch.stack(images, dim=0).numpy()
        assert images.shape == (len(image_paths), 2, *SHAPE) and images.dtype == np.uint8
        images = images.transpose(0, 1, 3, 4, 2)
        out_path = os.path.join(out_dir, f'{split}.npy')
        np.save(out_path, images)
        print(f'Created {split}.npy with shape {images.shape}')


    _process_files('train', 'KITTI_stereo_train.txt')
    _process_files('val', 'KITTI_stereo_val.txt')
    _process_files('test', 'KITTI_stereo_test.txt')

    print(f'KITTI Stereo successfully created.')


#class DISCUS(Dataset):
#    """Dataset for two correlated Gaussian sources used by DISCUS."""
#    def __init__(self, *, dim: int, sig_x: float=1.0, sig_n: float, split: str,
#                 length: int=50000, data_root=None):
#        assert split in ('train', 'val')
#        super().__init__()
#
#        self.dim = dim
#        self.sig_x = sig_x
#        self.sig_n = sig_n
#        self.split = split
#        self.length = length  # Fake length
#
#        if split == 'val':
#            rng = np.random.default_rng(1337)
#            self.val_x = torch.from_numpy(rng.standard_normal(size=[20000, dim], dtype=np.float32)) * sig_x
#            self.val_n = torch.from_numpy(rng.standard_normal(size=[20000, dim], dtype=np.float32)) * sig_n
#            self.length = 20000
#
#
#    def __getitem__(self, idx):
#        if self.split == 'train':
#            x = torch.randn(self.dim) * self.sig_x
#            n = torch.randn(self.dim) * self.sig_n
#        else:
#            x = self.val_x[idx]
#            n = self.val_n[idx]
#
#        return x, x + n
#
#    def __len__(self):
#        return self.length


class DISCUS(Dataset):
    """Dataset for two correlated Bernoulli sources used by DISCUS."""
    def __init__(self, *, dim: int, p_x: float, p_n: float, split: str,
                 length: int=50000, data_root=None):
        assert split in ('train', 'val')
        super().__init__()

        self.dim = dim
        self.p_x = p_x
        self.p_n = p_n
        self.split = split
        self.length = length  # Fake length

        if split == 'val':
            rng = np.random.default_rng(1337)
            self.val_x = torch.bernoulli(p_x * torch.ones((10000, dim)))
            self.val_n = torch.bernoulli(p_n * torch.ones((10000, dim)))
            self.length = 10000


    def __getitem__(self, idx):
        if self.split == 'train':
            x = torch.bernoulli(self.p_x * torch.ones(self.dim))
            n = torch.bernoulli(self.p_n * torch.ones(self.dim))
        else:
            x = self.val_x[idx]
            n = self.val_n[idx]

        return x, (x + n) % 2

    def __len__(self):
        return self.length


class MNIST(Dataset):
    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        if split == 'val':
            print(f'INFO: MNIST does not have an official validation set. Using test set instead.')
            split = 'test'

        self.split = split
        self.image_shape = (1, 28, 28)
        self.data_root = data_root

        self.data = np.load(os.path.join(self.data_root, f'mnist/{self.split}_data.npy'))
        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))
        self.label = np.load(os.path.join(self.data_root, f'mnist/{self.split}_label.npy'))
        self.label = torch.from_numpy(self.label)
        assert len(self.data) == len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class CIFAR10(Dataset):
    def __init__(self, *, split, data_root):
        self.split = split
        self.image_shape = (3, 32, 32)
        self.data_root = data_root

        if split  == 'train':
            self.data = np.load(os.path.join(self.data_root, 'cifar10/train.npy'))
        elif split == 'test':
            self.data = np.load(os.path.join(self.data_root, 'cifar10/test.npy'))
        else:
            raise ValueError(f'Invalid split: {split}')

        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# Old KittiStereo
class OldKittiStereo(Dataset):
    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        self.split = split
        self.image_shape = (3, 640, 1224)
        self.data_root = data_root
        self.data = np.load(os.path.join(self.data_root, f'kitti_stereo/{split}.npy'))
        self.data = torch.from_numpy(self.data)
        # self.data = torch.from_numpy(self.data.transpose(0,1,4,2,3))

    def __getitem__(self, idx):
        img = self.data[idx] # 2, H, W, C
        img = img.permute(3, 0, 1, 2) # C, 2, H, W
        img = img.reshape(3, 640, 1224)
        return img

    def __len__(self):
        return len(self.data)


class _KittiBase(Dataset):
    image_shape = None
    kitti_version = None

    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        assert self.kitti_version in ('KITTI_stereo', 'KITTI_general')
        assert len(self.image_shape) == 3 and self.image_shape[0] == 3
        self.split = split
        kitti_root = os.path.join(os.path.expanduser(data_root), 'kitti')
        path_file = os.path.join(kitti_root, f'{self.kitti_version}_{split}.txt')
        assert os.path.isfile(path_file), f'nonexistent path file: {path_file}'
        with open(path_file, 'r') as f:
            self.image_paths = [os.path.join(kitti_root, l.strip()) for l in f.readlines()]
        self.image_paths = list(zip(self.image_paths[0::2], self.image_paths[1::2]))
        assert len(self.image_paths) > 0

        # self.transforms = transforms.Compose([
        #     transforms.Lambda(_convert),
        #     transforms.CenterCrop([self.image_shape[1], self.image_shape[2]]),
        #     transforms.ToTensor(),
        #     transforms.Lambda(_tobyte),
        # ])
        self.transforms = [
            transforms.Lambda(_convert),
            transforms.CenterCrop([370, 740]),
            transforms.Resize([self.image_shape[1], self.image_shape[2]]),
            transforms.ToTensor(),
            transforms.Lambda(_tobyte),
        ]
        if self.split == 'train':
            self.transforms.append(transforms.RandomHorizontalFlip())
        self.transforms = transforms.Compose(self.transforms)

        # print(f'Creating KITTI dataset (split: {split}) with {len(self.image_paths)} images.')

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image1 = self.transforms(Image.open(image_path[0]))
        image2 = self.transforms(Image.open(image_path[1]))
        x = torch.cat([image1, image2], dim=1)
        return x

    def __len__(self):
        return len(self.image_paths)

        
class KittiStereo(_KittiBase):
    image_shape = (3, 128, 256)
    kitti_version = 'KITTI_stereo'


class _CelebAHQBase(Dataset):
    image_size = None

    def __init__(self, *, split, data_root):
        assert split in ('train', 'val', 'test'), f'Invalid split: {split}'
        if split == 'test':
            print(f'INFO: CelebaHQ does not have an official test set. Using val set instead.')
            split = 'val'

        self.split = split
        self.image_shape = (3, self.image_size, self.image_size)
        self.data_root = data_root
        self.data = np.load(os.path.join(data_root, f'celebahq/{split}_{self.image_size}x{self.image_size}.npy'))
        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class CelebAHQ32(_CelebAHQBase):
    image_size = 32


class CelebAHQ64(_CelebAHQBase):
    image_size = 64


class CelebAHQ128(_CelebAHQBase):
    image_size = 128


class CelebAHQ256(_CelebAHQBase):
    image_size = 256


class _OordImageNetBase(Dataset):
    image_size = None

    def __init__(self, *, split, data_root, mmap=True):
        self.split = split
        self.mmap = mmap
        self.image_shape = (3, self.image_size, self.image_size)
        self.data_root = data_root

        if split in ('train', 'valid'):
            fn = os.path.join(self.data_root, f'oord_imagenet{self.image_size}/{split}.npy')
            self.data = np.load(fn, mmap_mode=('r' if self.mmap else None))
        else:
            raise ValueError(f'Invalid split: {split}')

        self.data = torch.from_numpy(self.data.transpose(0,3,1,2))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class OordImageNet32(_OordImageNetBase):
    image_size = 32


class OordImageNet64(_OordImageNetBase):
    image_size = 64


class _ImageNetBase(Dataset):
    image_size = None

    def __init__(self, *, split, data_root):
        self.split = split
        self.image_shape = (3, self.image_size, self.image_size)

        if split == 'train':
            self.image_dir =  os.path.join(data_root, 'imagenet/train')
        elif split == 'valid':
            self.image_dir = os.path.join(data_root, 'imagenet/val')
        else:
            raise ValueError(f'Invalid split: {split}')

        self.transforms = transforms.Compose([
            transforms.Lambda(_convert),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Lambda(_tobyte),
        ])
        self.image_paths = [
            image_path for idx, image_path in
            enumerate(sorted(glob.glob(os.path.join(self.image_dir, '*.JPEG'))))
        ]
        assert len(self.image_paths) > 0

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        x = self.transforms(Image.open(image_path))
        assert x.shape == (3, self.image_size, self.image_size), \
                f'Wrong shape {x.shape} for ImageNet split={self.split} index={idx}'
        return x

    def __len__(self):
        return len(self.image_paths)


class ImageNet128(_ImageNetBase):
    image_size = 128


class ImageNet256(_ImageNetBase):
    image_size = 256


class Text8(Dataset):
    def __init__(self):
        raise NotImplementedError('Need to be updated to use pytorch Dataset!')
        self.all_data = torch.from_numpy(np.load(join(DATA_ROOT, 'text8/text8.npy'))).long()
        self.vocab = torch.from_numpy(np.load(join(DATA_ROOT, 'text8/vocab.npy')))
        self.train = self.all_data[:90000000]
        self.valid = self.all_data[90000000:90000000 + 5000000]
        self.test = self.all_data[90000000 + 5000000:]

    def to_strings(self, indices):
        assert indices.ndim == 2
        return [''.join(map(chr, row)) for row in self.vocab[indices.view(-1)].view_as(indices)]


##### Main methods #####

def prepare_dataset(dataset: str, data_root: str = _DEFAULT_DATA_ROOT):
    assert dataset is not None and data_root is not None
    mapping = {
        'mnist': _prepare_mnist,
        'text8': _prepare_text8,
        'cifar10': _prepare_cifar10,
        # 'kitti_stereo': _prepare_kitti_stereo,
        'celebahq32': functools.partial(_prepare_celebahq, size=32),
        'celebahq64': functools.partial(_prepare_celebahq, size=64),
        'celebahq128': functools.partial(_prepare_celebahq, size=128),
        'celebahq256': functools.partial(_prepare_celebahq, size=256),
        'oord_imagenet32': functools.partial(_prepare_oord_imagenet, size=32),
        'oord_imagenet64': functools.partial(_prepare_oord_imagenet, size=64),
    }
    if dataset not in mapping:
        raise ValueError(f'Invalid dataset name {dataset}')
    if dataset.startswith('celebahq'):
        out_dir = os.path.join(data_root, 'celebahq')
    else:
        out_dir = os.path.join(data_root, dataset)

    os.makedirs(out_dir, exist_ok=True)
    mapping[dataset](out_dir)


def load_dataset(dataset: str, *args, data_root: bool = None, **kwargs):
    if data_root is None:
        data_root = _DEFAULT_DATA_ROOT
    kwargs['data_root'] = data_root

    return {
        'discus': DISCUS,
        'text8': Text8,
        'mnist': MNIST,
        'cifar10': CIFAR10,
        'coco': CocoCaptions,
        'kitti_stereo': KittiStereo,
        'celebahq32': CelebAHQ32,
        'celebahq64': CelebAHQ64,
        'celebahq128': CelebAHQ128,
        'celebahq256': CelebAHQ256,
        'oord_imagenet32': OordImageNet32,
        'oord_imagenet64': OordImageNet64,
        'imagenet128': ImageNet128,
        'imagenet256': ImageNet256,
    }[dataset](*args, **kwargs)


##### TESTS #####


def test_strided_batch1d():
    data = torch.arange(20)
    B, T, S = 3, 7, 2
    xs, masks = zip(*strided_batch1d(data, batch_size=B, window_size=T, stride=S))
    assert len(xs) == 3
    assert xs[0].shape == xs[1].shape == (B, T)
    assert xs[2].shape == (2, T)
    xs = torch.cat(xs, dim=0)
    masks = torch.cat(masks, dim=0)
    assert (xs[masks] == torch.arange(len(data))).all()


def test_discus():
    sig_x, sig_n = 1.0, 0.2
    dataset = load_dataset('discus', dim=20000, sig_x=sig_x, sig_n=sig_n, split='train', length=12345)
    assert len(dataset) == 12345
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
    cnt = 0
    for x, y in loader:
        assert torch.isclose(x.std(), torch.tensor(sig_x), atol=1e-2, rtol=1e-2)
        assert torch.isclose(y.std()**2, torch.tensor(sig_x**2 + sig_n**2), atol=1e-2, rtol=1e-2)
        cnt += 1
        if cnt >= 10:
            break

    dataset = load_dataset('discus', dim=1000, sig_x=sig_x, sig_n=sig_n, split='val')
    assert len(dataset) == 20000
    x, y = zip(dataset[0], dataset[1], dataset[-2], dataset[-1])
    x = torch.stack(x)
    y = torch.stack(y)
    assert x.dtype == y.dtype == torch.float32 and x.shape == y.shape == (4, 1000)
    assert x.mean().item() == -0.026977768167853355 and y.mean().item() == -0.027465900406241417
    assert x.std().item() == 0.981647789478302 and y.std().item() == 1.008845567703247


def test_mnist():
    data = load_dataset('mnist', split='train')
    assert len(data) == 60000
    x, y = list(zip(data[0], data[1], data[-2], data[-1]))
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    assert x.dtype == torch.uint8 and x.shape == (4, 1, 28, 28) and x.sum().item() == 99968
    assert y.dtype == torch.int64 and y.shape == (4,) and y.tolist() == [5, 0, 6, 8]

    data = load_dataset('mnist', split='test')
    assert len(data) == 10000
    x, y = list(zip(data[0], data[1], data[-2], data[-1]))
    x = torch.stack(x, dim=0)
    y = torch.stack(y, dim=0)
    assert x.dtype == torch.uint8 and x.shape == (4, 1, 28, 28) and x.sum().item() == 115696
    assert y.dtype == torch.int64 and y.shape == (4,) and y.tolist() == [7, 2, 5, 6]


def test_cifar10():
    ds = load_dataset('cifar10', split='train')
    x = torch.stack([ds[0], ds[1], ds[-2], ds[-1]], dim=0)
    assert x.dtype == torch.uint8 and x.shape == (4, 3, 32, 32)
    assert x.float().mean().item() == 133.779541015625

    ds = load_dataset('cifar10', split='test')
    x = torch.stack([ds[0], ds[1], ds[-2], ds[-1]], dim=0)
    assert x.dtype == torch.uint8 and x.shape == (4, 3, 32, 32)
    assert x.float().mean().item() == 119.116943359375


def test_celebahq():
    dataset = load_dataset('celebahq64', split='full')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 30000
    assert x.shape == (4, 3, 64, 64) and x.dtype == torch.uint8
    assert x.sum() == 5037880

    dataset = load_dataset('celebahq128', split='full')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 30000
    assert x.shape == (4, 3, 128, 128) and x.dtype == torch.uint8
    assert x.sum() == 20168755


def test_kitti_stereo():
    dataset = load_dataset('kitti_stereo', split='train')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 1576
    assert x.shape == (4, 3, 256, 256) and x.dtype == torch.uint8
    assert x.sum() == 87569201


def test_oord_imagenet32():
    dataset = load_dataset('oord_imagenet32', split='train')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 1281149
    assert x.shape == (4, 3, 32, 32) and x.dtype == torch.uint8
    assert x.sum() == 1476073

    dataset = load_dataset('oord_imagenet32', split='valid')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 49999
    assert x.shape == (4, 3, 32, 32) and x.dtype == torch.uint8
    assert x.sum() == 1365737


def test_oord_imagenet64():
    dataset = load_dataset('oord_imagenet64', split='train')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 1281149
    assert x.shape == (4, 3, 64, 64) and x.dtype == torch.uint8
    assert x.sum() == 5319624

    dataset = load_dataset('oord_imagenet64', split='valid')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 49999
    assert x.shape == (4, 3, 64, 64) and x.dtype == torch.uint8
    assert x.sum() == 5447102


def test_imagenet128():
    # TODO: Add test for train set
    dataset = load_dataset('imagenet128', split='valid')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 50000
    assert x.shape == (4, 3, 128, 128) and x.dtype == torch.uint8
    assert x.sum() == 28558047


def test_imagenet256():
    # TODO: Add test for train set
    dataset = load_dataset('imagenet256', split='valid')
    x = torch.stack([dataset[0], dataset[1], dataset[-2], dataset[-1]], dim=0)
    assert len(dataset) == 50000
    assert x.shape == (4, 3, 256, 256) and x.dtype == torch.uint8
    assert x.sum() == 114236788


def test_text8():
    # TODO: Update this
    return
    text8 = load_dataset('text8')
    assert text8.train.shape == (90000000,)
    assert text8.valid.shape == text8.test.shape == (5000000,)
    assert text8.train.dtype == text8.valid.dtype == text8.test.dtype == torch.int64
    assert text8.all_data.shape == (100000000,)
    assert ''.join(map(chr, text8.vocab)) == ' abcdefghijklmnopqrstuvwxyz'

    x = torch.stack([text8.train[:100],text8.valid[:100],text8.test[:100]], dim=0).long()
    out = text8.to_strings(x)
    assert out[0] == ' anarchism originated as a term of abuse first used against early working class radicals including t'
    assert out[1] == 'e the capital of one government after another one of such governments was established in one eight s'
    assert out[2] == 'be ejected and hold it there examine the chamber to ensure it is clear allow the action to go forwar'

    assert text8.all_data.float().mean() == 9.700751304626465
    assert text8.train.float().mean() == 9.698159217834473
    assert text8.valid.float().mean() == 9.730711936950684
    assert text8.test.float().mean() == 9.71743392944336


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--data_root', type=str, default=_DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    if args.command == 'prepare':
       prepare_dataset(args.dataset, args.data_root)
    else:
       raise ValueError(f'Invalid command {args.command}')

