import argparse
import cv2


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sal_map_ver(v):
    if v.lower().replace(" ", "") in ('itti', 'koch', 'Itti'):
        return "itti"
    elif v.lower().replace(" ", "") in ('spectral', 'spectralresidual', 'sr', 'hou'):
        return "sr"
    elif v.lower().replace(" ", "") in ('bms', 'boolean', 'booleanmap', 'zhang'):
        return "bms"
    elif v.lower().replace(" ", "") in ('mbd', 'MBD'):
        return "mbd"
    elif v.lower().replace(" ", "") in ('rbd', 'RDB'):
        return "rbd"
    elif v.lower().replace(" ", "") in ('ft', 'FT'):
        return "ft"
    else:
        raise argparse.ArgumentTypeError('Incorrect value entered. Saliency map options are: "itti", "sr", "bms"')
