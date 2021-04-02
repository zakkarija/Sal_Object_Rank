import numpy as np
import cv2
# system imports
import timeit

# library imports
from skimage.color import gray2rgb, rgb2gray, rgb2lab
from skimage.io import imread, imsave
from skimage.transform import rescale


# -------------------------------------------------
# SpectralResidualSaliency Implementation
# from  https://github.com/uoip/SpectralResidualSaliency
# -------------------------------------------------

def SpectralResidualSaliency(img_col):
    img = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)

    WIDTH = int(img.shape[1])  # Altered from OG to keep same Width

    img = cv2.resize(img, (WIDTH, int(WIDTH * img.shape[0] / img.shape[1])))

    imgc = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(imgc[:, :, 0] ** 2 + imgc[:, :, 1] ** 2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

    imgc[:, :, 0] = imgc[:, :, 0] * spectralResidual / mag
    imgc[:, :, 1] = imgc[:, :, 1] * spectralResidual / mag
    imgc = cv2.dft(imgc, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = imgc[:, :, 0] ** 2 + imgc[:, :, 1] ** 2
    cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)

    return mag


# -------------------------------------------------
# SpectralResidualSaliency Implementation (mainly used for videos)
# from  https://github.com/mbeyeler/opencv-python-blueprints/blob/master/chapter5/saliency.py
# -------------------------------------------------

# Put saliency/SpectralResidualSaliency.py into root
# from SpectralResidualSaliency.py import Saliency

# def SpectralResidualSaliency2(img):
#     sal = Saliency(img, use_numpy_fft=True, gauss_kernel=(9, 9))
#     cv2.imshow('saliency', sal.get_saliency_map())
#     cv2.imshow('objects', sal.get_proto_objects_map(use_otsu=False))


# -------------------------------------------------
# Boolean Map Saliency Implementation
# from  https://github.com/fzliu/saliency-bms
# -------------------------------------------------

N_THRESHOLDS = 10


def activate_boolean_map(bool_map):
    """
        Performs activation on a single boolean map.
    """

    # use the boolean map as a mask for flood filling
    activation = np.array(bool_map, dtype=np.uint8)
    mask_shape = (bool_map.shape[0] + 2, bool_map.shape[1] + 2)
    ffill_mask = np.zeros(mask_shape, dtype=np.uint8)

    # top and bottom rows
    for i in range(0, activation.shape[0]):
        for j in [0, activation.shape[1] - 1]:
            if activation[i, j]:
                cv2.floodFill(activation, ffill_mask, (j, i), 0)

    # left and right columns
    for i in [0, activation.shape[0] - 1]:
        for j in range(0, activation.shape[1]):
            if activation[i, j]:
                cv2.floodFill(activation, ffill_mask, (j, i), 0)

    return activation


def compute_saliency(img):
    """
        Computes Boolean Map Saliency (BMS).
    """

    img_lab = rgb2lab(img)
    img_lab -= img_lab.min()
    img_lab /= img_lab.max()
    thresholds = np.arange(0, 1, 1.0 / N_THRESHOLDS)[1:]

    # compute boolean maps
    bool_maps = []
    for thresh in thresholds:
        img_lab_T = img_lab.transpose(2, 0, 1)
        img_thresh = (img_lab_T > thresh)
        bool_maps.extend(list(img_thresh))

    # compute mean attention map
    attn_map = np.zeros(img_lab.shape[:2], dtype=np.float)
    for bool_map in bool_maps:
        attn_map += activate_boolean_map(bool_map)
    attn_map /= N_THRESHOLDS

    # gaussian smoothing
    attn_map = cv2.GaussianBlur(attn_map, (0, 0), 3)

    # perform normalization
    norm = np.sqrt((attn_map ** 2).sum())
    attn_map /= norm
    attn_map /= attn_map.max() / 255

    return attn_map.astype(np.uint8)


def bms(img):
    """
        Entry point.
    """

    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    upper_dim = max(img.shape[:2])
    # For dim
    # if upper_dim > args.max_dim:
    #     img = rescale(img, args.max_dim / float(upper_dim), order=3)

    # compute saliency
    return compute_saliency(img)