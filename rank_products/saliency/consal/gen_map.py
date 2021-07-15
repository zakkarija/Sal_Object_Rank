import cv2
import os
import saliency.consal.main as main


def sal_eval(img_path):
    sal_path = main.eval_image(img_path)

    if not os.path.exists(sal_path):
        print("Saliency Map created has not been found!!")

    return cv2.imread(sal_path)

