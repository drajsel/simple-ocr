import cv2
import numpy as np

from ocr.segmentation import find_contours


def autocrop(image, contour_mode=False):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if contour_mode:
        contours = find_contours(image)
        top = np.inf
        bottom = -1
        left = np.inf
        right = -1

        for cnt in contours:
            x, y, w, h = cnt
            if x < left:
                left = x
            if x + w > right:
                right = x + w
            if y < top:
                top = y
            if y + h > bottom:
                bottom = y + h

        if left == np.inf:
            left = 0
        if top == np.inf:
            top = 0
        if bottom < 0:
            bottom = image.shape[0]
        if right < 0:
            right = image.shape[1]

        return image[top:bottom, left:right]

    else:
        top = 0
        bottom = image.shape[0]
        left = 0
        right = image.shape[1]

        for i in range(image.shape[0]):
            black_pixels = np.where(image[i, :] < 255)
            if len(black_pixels):
                top = i
                break
        bottom = image.shape[0]
        for i in reversed(range(image.shape[0])):
            black_pixels = np.where(image[i, :] < 255)
            if len(black_pixels):
                bottom = i
                break
        for i in range(image.shape[1]):
            black_pixels = np.where(image[:, i] < 255)
            if len(black_pixels):
                left = i
                break
        for i in reversed(range(image.shape[1])):
            black_pixels = np.where(image[:, i] < 255)
            if len(black_pixels):
                right = i
                break

        return image[top:bottom, left:right]


def display_rect(image, rect, color=(0, 0, 255), thickness=1):
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def remove_noise(img, threshold=0.1, pad=0, blur_size=1):
    """
    Function for removing objects on the border, for example, because the image was not cropped correctly.

    :param img: input image
    :param threshold: it is assumed that the objects want to delete can be found in threshold % of the image in
                            each dimension
    :param pad: number of pixels by which the image was padded in each direction

    """
    original = img.copy()
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.blur(img, (blur_size, blur_size), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (y <= pad + 1 and y + h <= threshold * (img.shape[0] + 2 * pad)) \
                or y >= (1.0 - threshold) * (2 * pad + img.shape[0]):
            # or (x <= pad + 1 and x + w <= threshold * (2*pad + img.shape[1]))\
            # or x >= (1.0 - threshold)*(2*pad + img.shape[1]):

            original[y:y + h, x:x + w] = 255

    return original


def add_padding_to_characters(char_img, eps=0.25):
    """
    We have to somehow differentiate lowercase and uppercase letters. Otherwise, c/C and z/Z are sometimes
    misclassified.
    For the approach to be robust to scaling, the padding depends only on the shape of the character.
    We add padding only to letters that are (almost) quadratic in size, i.e. lowecase c, s, ... Not to l, j, i...
    Value of eps was found empirically.

    :param char_img: input image containing the character
    :param eps: the threshold determining the ratio based on which it is decided whether or not to add padding to the
                character

    :return char_img: return (padded) character image
    """

    h, w = char_img.shape[:2]
    # add top padding to character if the width and height are in certain ratios
    if (1.0 - eps) * w <= h <= (1.0 + eps) * w:
        return cv2.copyMakeBorder(char_img, top=int(0.35 * h), bottom=0, left=0,  # top=w
                                  right=0, borderType=cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
    else:
        return char_img