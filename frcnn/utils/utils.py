import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import logging
from frcnn.utils.log import logger
import xml.etree.cElementTree as ET
from tensorflow.python.tools import inspect_checkpoint as chkp

logger = logging.getLogger('logger')

def load_image(url, show_info=False):
    img = plt.imread(url)
    img = np.array(img / 255, dtype=np.float32)
    if show_info:
        logger.info('Loading image from \'{}\''.format(url))
    return img

def save_image(url, img, show_info=False):
    plt.imsave(url, img)
    if show_info:
        logger.info('Saving image to \'{}\''.format(url))
    return url

def save_plot(url, f, show_info=False):
    f.savefig(url)
    if show_info:
        logger.info('Saving plot to \'{}\''.format(url))
    return url

def save_npy(url, arr, show_info=False):
    np.save(url, arr)
    if show_info:
        logger.info('Saving npy to \'{}\''.format(url))

def load_npy(url, show_info=False):
    data = np.load(url)
    if show_info:
        logger.info('Loading npy from \'{}\''.format(url))
    return data

def load_txt(url, show_info=False):
    data = [i.replace('\n', '') for i in open(url).readlines()]
    if show_info:
        logger.info('Loading text from \'{}\''.format(url))
    return data

def load_xml(url, show_info=False):
    xml = ET.parse(url)
    root = xml.getroot()
    if show_info:
        logger.info('Loading XML from {}'.format(url))
    return root

def print_xml(url):
    root = load_xml(url)
    for c1 in root:
        print('{}:{}'.format(c1.tag, c1.text))
        for c2 in c1:
            print('  |__{}:{}'.format(c2.tag, c2.text))
            for c3 in c2:
                print('    |__{}:{}'.format(c3.tag, c3.text))
                for c4 in c3:
                    print('      |__{}:{}'.format(c4.tag, c4.text))

def parse_object(url):
    root = load_xml(url)
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    depth = int(root.find('size').find('depth').text)
    objects = root.findall('object')
    obj_infos = []
    for object in objects:
        name = object.find('name').text
        bndbox = ' '.join([i.text for i in object.find('bndbox')])
        obj_infos.append([name, bndbox])
    return root.find('filename').text, width, height, depth, obj_infos

def parse_part():
    pass

def correct_bbox(shape, bbox):
    W, H = shape
    x, y, w, h = bbox
    x = np.minimum(np.maximum(0, x), W)
    y = np.minimum(np.maximum(0, y), H)
    if x + w >= W:
        w = W - x
    if y + h >= H:
        h = H - y
    return x, y, w, h

def annotate_image(image, bbox):
    '''
    Annotate image with bbox
    :param image: url or numpy.ndarray or list
    :param bbox: list or numpy.ndarray with shape (?, 4)
    :return: plot of annotated image
    '''
    if isinstance(image, str):
        try:
            image = load_image(image)
        except:
            logger.error('Cannot load image !')
            raise IOError
    image = np.array(image)
    bbox = np.array(bbox)
    if len(bbox.shape) != 2 or bbox.shape[1] != 4:
        logger.error('Expect bbox have shape (?, 4) but got {} !'.format(bbox.shape))
        raise ValueError

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image)
    for b in bbox:
        b = correct_bbox(image.shape[:-1], b)
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, facecolor='none', edgecolor='r'))
    ax.set_axis_off()
    return f

def print_model(ckpt_path, tensor_name=None, all_tensors=False, all_tensor_names=True):
    chkp.print_tensors_in_checkpoint_file(file_name=ckpt_path,
                                          tensor_name=tensor_name,
                                          all_tensors=all_tensors,
                                          all_tensor_names=all_tensor_names)