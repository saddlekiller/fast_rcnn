from frcnn.utils.utils import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    image = load_image('images/test_image.jpg')
    bbox = np.array(np.random.random((50, 4)) * 200, np.int32)
    f = annotate_image(image, bbox)
    save_plot('images/test_annotate.jpg', f)
    save_image('images/test_save_image.jpg', image)
