from easydict import EasyDict
import os
import copy

hparams = EasyDict()
hparams.dataset = EasyDict()
hparams.dataset.name = 'ILSVRC'

if hparams.dataset.name == 'VOC2007':
    hparams.dataset.base_dir = os.environ['VOC2007']
    hparams.dataset.train_dir = hparams.dataset.base_dir + '/VOCtrainval'
    hparams.dataset.valid_dir = hparams.dataset.base_dir + '/VOCtest'

    hparams.train_preproc = EasyDict()
    hparams.train_preproc.base_dir = hparams.dataset.train_dir + '/VOC2007'
    hparams.train_preproc.annotations_dir = hparams.train_preproc.base_dir + '/Annotations'
    hparams.train_preproc.imagesets_dir = hparams.train_preproc.base_dir + '/ImageSets'
    hparams.train_preproc.jpegimages_dir = hparams.train_preproc.base_dir + '/JPEGImages'
    hparams.train_preproc.data_dir = hparams.train_preproc.base_dir + '/data'
    hparams.train_preproc.image_dir = hparams.train_preproc.data_dir + '/images'
    hparams.train_preproc.metadata = hparams.train_preproc.data_dir + '/metadata'
    hparams.train_preproc.objects = hparams.train_preproc.data_dir + '/objects'

    hparams.valid_preproc = EasyDict()
    hparams.valid_preproc.base_dir = hparams.dataset.valid_dir + '/VOC2007'
    hparams.valid_preproc.annotations_dir = hparams.valid_preproc.base_dir + '/Annotations'
    hparams.valid_preproc.imagesets_dir = hparams.valid_preproc.base_dir + '/ImageSets'
    hparams.valid_preproc.jpegimages_dir = hparams.valid_preproc.base_dir + '/JPEGImages'
    hparams.valid_preproc.data_dir = hparams.valid_preproc.base_dir + '/data'
    hparams.valid_preproc.image_dir = hparams.valid_preproc.data_dir + '/images'
    hparams.valid_preproc.metadata = hparams.valid_preproc.data_dir + '/metadata'
    hparams.valid_preproc.objects = hparams.valid_preproc.data_dir + '/objects'

if hparams.dataset.name == 'ILSVRC':
    hparams.dataset.base_dir = os.environ['ILSVRC']
    hparams.dataset.Annotations = hparams.dataset.base_dir + '/Annotations'
    hparams.dataset.Data = hparams.dataset.base_dir + '/Data'
    hparams.dataset.ImageSets = hparams.dataset.base_dir + '/ImageSets'

    hparams.train_preproc = EasyDict()
    hparams.train_preproc.base_dir = hparams.dataset.base_dir
    hparams.train_preproc.Annotations = hparams.dataset.Annotations + '/CLS-LOC/train'
    hparams.train_preproc.Data = hparams.dataset.Data + '/CLS-LOC/train'
    hparams.train_preproc.ImageSets = hparams.dataset.ImageSets + '/CLS-LOC/train'
    hparams.train_preproc.metadata = hparams.dataset.base_dir + 'train_metadata'

    hparams.valid_preproc = EasyDict()
    hparams.valid_preproc.base_dir = hparams.dataset.base_dir
    hparams.valid_preproc.Annotations = hparams.dataset.Annotations + '/CLS-LOC/val'
    hparams.valid_preproc.Data = hparams.dataset.Data + '/CLS-LOC/val'
    hparams.valid_preproc.ImageSets = hparams.dataset.ImageSets + '/CLS-LOC/val'
    hparams.valid_preproc.metadata = hparams.dataset.base_dir + 'val_metadata'





