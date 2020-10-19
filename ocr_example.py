import cv2
import time
import os
import argparse

from ocr.classification import classify

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--input",
    required=True,
    help="path to the image on which OCR will be performed"
)
ap.add_argument(
    "-f", "--font",
    required=True,
    help="Name of the font used (as stored in the 'fonts' directory."
)
ap.add_argument(
    "-s", "--show_steps",
    default=False,
    type=bool,
    help="Whether or not to display steps of the OCR process."
)

args = vars(ap.parse_args())


if __name__ == "__main__":
    img_path = args['input']
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("No such image. Check the file path!")
        exit(-1)

    n = 100
    exec_time = 0.0
    for _ in range(n):
        start = time.time()
        classify(img, show_steps=args['show_steps'], feature_size=50,
                 font_name=args['font'])
        exec_time += time.time() - start

    print(classify(img, show_steps=args['show_steps'], feature_size=50,
                   font_name=args['font']))
    print("Average classification time ({} it): {} ms".format(n, int(1000 * (exec_time / n))))
