import numpy as np
import os
from frcnn.configs.hparams import hparams
from frcnn.utils.utils import *

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
        

    def _sampling(self, batch_size):
        return [self.metadata[np.random.randint(0, self.n_samples)] for i in range(batch_size)]

    def _load_meta(self):
        self.metadata = load_txt(self.metadata_path)
        self.objectset = load_txt(self.objects_path)
        self.n_samples = len(self.metadata)

if __name__ == '__main__':
    # lines = open(hparams.train_preproc.metadata).readlines()
    # for line in lines:
    #     print(parse_meta(line))
    #     break
    feeder = VOCFeeder(hparams.train_preproc)
    print(feeder._prepare_batch(2))