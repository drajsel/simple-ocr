import numpy as np
import cv2

MIN_CONTOUR_AREA = 10
BLUR_SIZE = 1


def display_rect(img, rect, color=(0,0,255), thickness=1):
    x, y, w, h = rect
    cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)


def find_contours(thresh_image, offset_x=0, offset_y=0):
    # introducing offset to "findContours" because the rectangles tend to be lower than the original
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                           offset=(offset_x, offset_y))
    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [X, Y, W, H] = cv2.boundingRect(contour)
            bounding_boxes.append((X,Y,W,H))

    return bounding_boxes


def region_from_segment(image, segment):
    """given a segment (rectangle) and an image, returns it's corresponding subimage"""
    x, y, w, h = segment
    if len(image.shape) == 2:
        return image[y:y + h, x:x + w]
    else:
        return image[y:y + h, x:x + w, :]


def process_image(img, show_steps=False, offset_y=0, offset_x=0):
    line_segments = [] # 2d array because of lines
    line_imgs = []

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.blur(img, (BLUR_SIZE, BLUR_SIZE), 0)
    #blurred = cv2.GaussianBlur(img, (BLUR_SIZE, BLUR_SIZE), cv2.BORDER_DEFAULT)
    img_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 1. detect rectangles before line and character detection to estimate the font size
    characters = find_contours(img_thresh)
    max_char_height = max([rect[3] for rect in characters])
    max_char_width = max([rect[2] for rect in characters])

    # print("Maximum height: ", max_char_height)
    # print("Maximum width: ", max_char_width)

    # 2. separate text into lines
    rect_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3*max_char_width, int(0.2*max_char_height)))
    threshed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, rect_line_kernel)

    if show_steps:
        cv2.imshow("line morphing", threshed)
        cv2.waitKey(0)

    lines = find_contours(threshed)

    # 3. separate each line into characters
    for line in lines:
        line_img = region_from_segment(img, line)
        line_img = cv2.copyMakeBorder(line_img, top=max_char_height, bottom=max_char_height,
                                      left=max_char_height, right=max_char_height, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))
        line_imgs.append(line_img)

        if show_steps:
            cv2.imshow("detected line", line_img)
            cv2.waitKey(0)

        line_img_thresh = cv2.copyMakeBorder(region_from_segment(img_thresh, line),
                                             top=max_char_height, bottom=max_char_height, left=max_char_height,
                                             right=max_char_height,
                                             borderType=cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
        if show_steps:
            cv2.imshow("line threshed", line_img_thresh)
            cv2.waitKey(0)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                (1, int(1.5 * max_char_height / 5)))  # before max(heights)/5
        threshed = cv2.morphologyEx(line_img_thresh, cv2.MORPH_CLOSE, rect_kernel)
        if show_steps:
            cv2.imshow("character morphing", threshed)
            cv2.waitKey(0)

        characters = find_contours(threshed, offset_y=offset_y, offset_x=offset_x)

        if show_steps:
            char_img = line_img.copy()
            for char in characters:
                display_rect(char_img, char)
            cv2.imshow("detected characters", char_img)
            cv2.waitKey(0)

        segments = np.array(characters)

        # sort characters by x-coordinate
        segments = segments[segments[:,0].argsort()]
        line_segments.append(segments)

    lines = np.array(lines)
    line_segments = np.array(line_segments, dtype=object)

    # sort lines by y-coordinate
    ind = lines[:,1].argsort()
    line_segments = line_segments[ind]
    line_imgs = np.array(line_imgs, dtype=object)[ind]

    return line_segments, line_imgs
