import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import logging
from frcnn.utils.log import logger
import xml.etree.cElementTree as ET
from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
import itertools
from frcnn.dataset.feeder import *
import cv2

logger = logging.getLogger('logger')

def load_image(url, show_info=False, norm = False):
    img = plt.imread(url)
    if norm:
        img = np.array(img / 255, dtype=np.float32)
    else:
        img = np.array(img, dtype=np.float32)
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

def annotate_image(image, bbox, corner = True, norm=False):
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
    if norm:
        image /= 255
    bbox = np.array(bbox)
    if len(bbox.shape) != 2 or bbox.shape[1] != 4:
        logger.error('Expect bbox have shape (?, 4) but got {} !'.format(bbox.shape))
        raise ValueError

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(image)
    for b in bbox:
        if corner:
            b[2] -= b[0]
            b[3] -= b[1]
        b = correct_bbox(image.shape[:-1][::-1], b)
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, facecolor='none', edgecolor='r'))
    ax.set_axis_off()
    return f

def print_model(ckpt_path, tensor_name=None, all_tensors=False, all_tensor_names=True):
    chkp.print_tensors_in_checkpoint_file(file_name=ckpt_path,
                                          tensor_name=tensor_name,
                                          all_tensors=all_tensors,
                                          all_tensor_names=all_tensor_names)

def resize_image(image, shape, func=tf.image.resize_bilinear):
    return func(image, shape)

def resize_shape_inference(fshape, min_shape):
    H, W, _ = fshape
    ms = np.min([H, W])
    if H < W:
        ms = H
        ratio = min_shape / ms
        rw = ratio * W
        rh = min_shape
        newshape = np.array([rh, rw], dtype=np.int32)
    else:
        ms = W
        ratio = min_shape / ms
        rw = min_shape
        rh = ratio * H
        newshape = np.array([rh, rw], dtype=np.int32)
    return tuple(newshape)

def _anchor(scale, ratio):
    m = len(scale)
    n = len(ratio)
    anchors = [None] * (n * m)
    for i, s in enumerate(scale):
        for j, r in enumerate(ratio):
            anchors[i * n + j] = [0, 0, int(s / r), int(s * r)]
    return np.array(anchors)

def anchor(loc, anchors):
    res = [None] * len(anchors)
    loc = [loc[0], loc[1], loc[0], loc[1]]
    for i in range(len(anchors)):
        res[i] = list(np.array(loc) + np.array(anchors[i]))
    return res

def clip_bounder(bbox, shape):
    H, W, _ = shape
    xmin, ymin, xmax, ymax = bbox
    xmin = np.maximum(xmin, 0)
    xmax = np.minimum(xmax, W)
    ymin = np.maximum(ymin, 0)
    ymax = np.minimum(ymax, H)
    if xmin == xmax:
        return None
    if ymin == ymax:
        return None
    return [xmin, ymin, xmax, ymax]

def generate_anchors(shape, scales, ratios):
    anchors = []
    anchors_root = _anchor(scales, ratios)
    H, W, _ = shape
    for x in range(W):
        for y in range(H):
            temp = anchor((x,y), anchors_root)
            temp = [clip_bounder(i, shape) for i in temp if clip_bounder(i, shape) is not None]
            for t in temp:
                if t not in anchors:
                    anchors.append(t)
    return np.array(anchors)

def adaptive_pooling(inputs, output_shape):
    pass

def proposal_extraction(input_shape, pooling_scale, scales, ratios):
    pass

if __name__ == '__main__':
    from frcnn.configs.hparams import *
    # print(hparams)
    # feeder = VOCFeeder(hparams.train_preproc)
    # data = feeder._prepare_batch(1)
    # img = data[0][0]
    # # shape = data[3][0]
    # # print(shape)
    # # shapes = [[500, 328, 3], [300, 500, 3]]
    # # for shape in shapes:
    # #     print(resize_shape_inference(shape, 600))
    # # img = tf.constant(img, dtype=tf.float32)
    # # _img = resize_image(img, (600, 1000))
    # a = _anchor([32, 64, 128], [0.5, 1, 2])
    # b = anchor((100, 200), a)
    # print(img.shape, data[3])
    # annotate_image(img, b, norm=True)
    # plt.show()

    #
    b = generate_anchors((25, 18, 3), [4, 8, 16], [0.5, 1, 2])
    print(len(b))
        # with tf.Session() as sess:
    #     a, b = sess.run([img, _img])
    # print(data[1], data[2], data[3])
    # f = plt.figure()
    # ax = f.add_subplot(211)
    # ax.imshow(a[0] / 255)
    # ax = f.add_subplot(212)
    # ax.imshow(b[0] / 255)
    # plt.show()