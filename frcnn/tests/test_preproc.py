import numpy as np
from frcnn.dataset.preproc import VOC2007Preprocess
import xml.etree.cElementTree as ET
import os
from frcnn.configs.hparams import hparams
from frcnn.utils.utils import *

if __name__ == '__main__':
    url = 'xmls/test.xml'
    preproc = VOC2007Preprocess(hparams.valid_preproc)
    preproc.run()

    # annotations_dir = hparams.feeder.annotations_dir
    # urls = [annotations_dir + '/' + i for i in os.listdir(annotations_dir) if i.index('.xml') != -1]
    # for i in urls:
    #     print_xml(i)
    #     break






