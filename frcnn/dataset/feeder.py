import numpy as np
import os
from frcnn.configs.hparams import hparams
from frcnn.utils.utils import *
import PIL.Image as Image

def parse_meta(meta):
    meta = meta.replace('\n', '')
    filename, width, height, depth, label, bndbox = meta.split('|')
    width = int(width)
    height = int(height)
    depth = int(depth)
    bndbox = [int(i) for i in bndbox.split()]
    image = load_npy(filename)
    return image, [width, height, depth], label, bndbox

class VOCFeeder():

    def __init__(self, hparams):
        self.metadata_path = hparams.metadata
        self.objects_path = hparams.objects
        self.metadata = None
        self.n_samples = 0
        self.objectset = None
        self._load_meta()

    def _prepare_batch(self, batch_size):
        if batch_size != 1:
            batch_size = 1
            logger.warning('batch size > 1 is not supported yet, reseted to 1')
        meta = self._sampling(batch_size)
        batch_image = [None] * batch_size
        batch_label = [None] * batch_size
        batch_bndbox = [None] * batch_size
        batch_shape = [None] * batch_size
        for i, m in enumerate(meta):
            image, shape, label, bndbox = parse_meta(m)
            batch_image[i] = image
            batch_label[i] = self.objectset.index(label)
            batch_bndbox[i] = bndbox
            batch_shape[i] = shape
        batch_image = np.array(batch_image)
        batch_label = np.array(batch_label)
        batch_bndbox = np.array(batch_bndbox)
        batch_shape = np.array(batch_shape)
        return (batch_image, batch_label, batch_bndbox, batch_shape)

    def _sampling(self, batch_size):
        return [self.metadata[np.random.randint(0, self.n_samples)] for i in range(batch_size)]

    def _load_meta(self):
        self.metadata = load_txt(self.metadata_path)
        self.objectset = load_txt(self.objects_path)
        self.n_samples = len(self.metadata)

class TinyImageNetFeeder(object):

    def __init__(self, shape):
        self.base_dir = '/data/tiny-imagenet/tiny-imagenet-200'
        self.train_dir = self.base_dir + '/train'
        self.val_dir = self.base_dir + '/val'
        self.test_dir = self.base_dir + '/test'
        self.shape = shape

    def run(self, batch_size):
        labels = os.listdir(self.train_dir)
        for label in labels:
            image_dirs = os.listdir(os.path.join(self.train_dir, label, 'images'))[:32]
            n_images = len(image_dirs)
            n_batches = n_images // batch_size + 1
            image_dirs = [os.path.join(self.train_dir, label, 'images', i) for i in image_dirs]
            for i in range(n_batches):
                yield label, self._prepare_batch(image_dirs[i*batch_size:(i+1)*batch_size])

    def _prepare_batch(self, dirs):
        imgs = np.zeros((len(dirs), self.shape[0], self.shape[1], 3))
        for i, dir in enumerate(dirs):
            imgs[i] = self._read_image(dir)
        return np.array(imgs)

    def _reshape(self, img):
        return img.resize(self.shape)

    def _read_image(self, dir):
        img = Image.open(dir)
        img = self._reshape(img)
        img = np.array(img)
        if len(img.shape) != 3:
            img = np.expand_dims(img, -1)
            img = np.tile(img, [1, 1, 3])
        img = np.array(img) / 255
        return img

if __name__ == '__main__':
    # lines = open(hparams.train_preproc.metadata).readlines()
    # for line in lines:
    #     print(parse_meta(line))
    #     break
    # feeder = VOCFeeder(hparams.train_preproc)
    # print([i.shape for i in feeder._prepare_batch(2)])

    feeder = TinyImageNetFeeder((224, 224))
    iter = feeder.run(90)
    for i, j in iter:
        print(i, j.shape)
