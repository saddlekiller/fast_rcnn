from easydict import EasyDict
import os
import copy

hparams = EasyDict()
hparams.dataset = EasyDict()
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





