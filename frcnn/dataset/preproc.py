from frcnn.utils.utils import *
import numpy as np
import xml.etree.cElementTree as ET
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from functools import partial
from frcnn.utils.log import Logger
import logging
import argparse

logger = logging.getLogger('logger')

class VOC2007Preprocess():

    def __init__(self, hparams):
        self.base_dir = hparams.base_dir
        self.annotations_dir = hparams.annotations_dir
        self.imagesets_dir = hparams.imagesets_dir
        self.jpegimages_dir = hparams.jpegimages_dir
        self.data_dir = hparams.data_dir
        self.image_dir = hparams.image_dir
        self.metadata = hparams.metadata
        self.objects = hparams.objects
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def run(self):
        if os.path.exists(self.metadata):
            logger.warning('{} exists. Delete first if want to re-generate'.format(self.metadata))
        else:
            self.gen_image()
            self.gen_meta()

    def gen_meta(self):
        executor = ProcessPoolExecutor(max_workers=cpu_count())
        xml_urls = [i for i in os.listdir(self.annotations_dir) if i.index('.xml') != -1 ]
        meta = []
        for url in xml_urls:
            meta.append(executor.submit(partial(self._gen_meta, self.annotations_dir + '/' + url)))
        metas = [future.result() for future in tqdm(meta) if future.result() is not None]
        labels = set()
        with open(self.metadata, 'w') as f:
            for m in metas:
                for mi in m:
                    labels.add(mi.split('|')[-2])
                    f.write(mi + '\n')
        logger.info('meta data has been written in file \'{}\''.format(self.metadata))
        with open(self.objects, 'w') as f:
            for i in list(labels):
                f.write(i + '\n')
        logger.info('labels have been written in file \'{}\''.format(self.objects))
        return metas

    def gen_image(self):
        image_urls = [i for i in os.listdir(self.jpegimages_dir) if i.index('.jpg') != -1 ]
        executor = ProcessPoolExecutor(max_workers=cpu_count())
        image_meta = []
        for url in tqdm(image_urls):
            image_meta.append(executor.submit(partial(self._gen_image, self.jpegimages_dir + '/' + url)))
        logger.info('image meta data has been generated successfully')
        images = [future.result() for future in tqdm(image_meta) if future.result() is not None]
        for url, image in tqdm(images):
            save_npy(url, image)
        logger.info('pre-processed images has been saved in path \'{}\''.format(self.image_dir))

    def _gen_meta(self, url):
        filename, width, height, depth, obj_infos = parse_object(url)
        metadata = [None] * len(obj_infos)
        for i, obj_info in enumerate(obj_infos):
            metadata[i] = '|'.join([self.image_dir + '/' + filename.replace('.jpg', '.npy'), str(height), str(width), str(depth), '|'.join(obj_info)])
        return metadata

    def _gen_image(self, url):
        image = load_image(url)
        image_name = url.split('/')[-1]
        image_name = self.image_dir + '/' + image_name.replace('.jpg', '.npy')
        return image_name, image

class ILSVRCPreprocess():

    def __init__(self, hparams):
        self.base_dir = hparams.base_dir
        self.annotations_dir = hparams.Annotations
        self.data_dir = hparams.Data
        self.imagesets = hparams.ImageSets
        self.metadata = hparams.metadata

    def _process(self):
        pass

    def _retrieve_urls(self):
        urls = []
        for label in os.listdir(self.annotations_dir):
            for file in os.listdir(self.annotations_dir + '/' + label):
                url = os.path.join(self.annotations_dir, label, file)
                # print(url)
                urls.append(url)
                # break
            # break
        return urls

    def run(self):
        urls = self._retrieve_urls()
        for url in urls:
            meta = self._gen_meta(url)
            print(url)
            for m in meta:
                i = m.split('|')[0]
                img = load_image(i)

    def _gen_meta(self, url):
        filename, width, height, depth, obj_infos = parse_object(url)
        metadata = [None] * len(obj_infos)
        for i, obj_info in enumerate(obj_infos):
            metadata[i] = '|'.join([self.data_dir + '/' + filename.split('_')[0] + '/' + filename + '.JPEG', str(width), str(height), str(depth), '|'.join(obj_info)])
        return metadata


def get_arguments():
    logger.hline()
    logger.info('         ___              __                                     ')
    logger.info('        / __\            / /                                     ')
    logger.info('      _/ / ____   ___  _/ / ___   ____ ____ __   _   __ _   __   ')
    logger.info('     /  _// __ \ / _ \/  _// _ \ / __// __//  \ / | / // | / /   ')
    logger.info('     / / / / / / \ \\\\// / / ___// /  / /  / /_//  |/ //  |/ /    ')
    logger.info('    / / / /_/ //\/ / / /_/ /__ / /  / /  / /_ / /|  // /|  /     ')
    logger.info('   /_/  \___,/ \__/  \__/\___//_/  /_/   \__//_/ |_//_/ |_/      ')
    logger.hline()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hparams', type=str, default='train_preproc', help='')
    return parser.parse_args()

def main():
    args = get_arguments()
    from frcnn.configs.hparams import hparams
    dataset_name = hparams.dataset.name
    hparams = getattr(hparams, args.hparams)
    if dataset_name == 'VOC2007':
        preproc = VOC2007Preprocess(hparams)
    else:
        preproc = ILSVRCPreprocess(hparams)
    preproc.run()

if __name__ == '__main__':
    main()


#

#
#
#
#
#
#
#


