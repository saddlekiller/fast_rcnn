from easydict import EasyDict
import os

hparams = EasyDict()
hparams.feeder = EasyDict()
hparams.feeder.base_dir = os.environ['FRCNN_DATA']
hparams.feeder.annotations_dir = hparams.feeder.base_dir + '/Annotations'
hparams.feeder.imagesets_dir = hparams.feeder.base_dir + '/ImageSets'
hparams.feeder.jpegimages_dir = hparams.feeder.base_dir + '/JPEGImages'
