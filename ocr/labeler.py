import cv2
import os
import matplotlib.pyplot as plt
import string

from ocr.segmentation import process_image, region_from_segment, autocrop
from utils.visualisation_utils import draw_classes, draw_segments, show_image_and_wait_for_key
from six import unichr

ALLOWED_CHARS = list(map(ord, string.digits + string.ascii_letters + string.punctuation))
PADDING = 30


def label(image_filename, font_name, input_mode="matplotlib", folder_name="train_data"):
    img = cv2.imread("data/{}".format(image_filename))

    if img is None:
        img = cv2.imread("{}".format(image_filename))
        if img is None:
            print("Wrong image path")
            exit(0)

    cv2.imshow("Training data", img)

    img = cv2.copyMakeBorder(img, top=PADDING, bottom=PADDING, left=PADDING, right=PADDING,
                             borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
    line_segments, line_imgs = process_image(img.copy(), show_steps=False, offset_y=0, offset_x=0)

    save_path = os.path.join("data", "fonts", font_name, folder_name)
    os.makedirs(save_path, exist_ok=True)
    ind = -1

    with open("{}/labels.txt".format(save_path), "w") as f:
        output_classes = []
        output_images = []
        if input_mode == "matplotlib":
            for line_segment, line_img in zip(line_segments, line_imgs):
                for i in range(len(line_segment)):
                    ind += 1

                    char_img = region_from_segment(line_img, line_segment[i])
                    char_img = autocrop(char_img)
                    cv2.imwrite("{}/{}.png".format(save_path, ind), char_img)

                    plt.imshow(char_img)
                    plt.show()
                    plt.pause(0.001)

                    char = input("Input character {}:\n".format(ind))
                    print("Detected: ", char)
                    f.write("{}.png {}\n".format(ind, char))

        elif input_mode == "cv2":
            for line_segment, line_img in zip(line_segments, line_imgs):
                classes = ['??' for _ in range(len(line_segment))]
                images = [None for _ in range(len(line_segment))]
                i = 0

                while True:
                    if i < 0:
                        i = 0
                    elif i >= len(line_segment):
                        i = len(line_segment) - 1

                    char_img = region_from_segment(line_img, line_segment[i])
                    char_img = autocrop(char_img)
                    images[i] = char_img
                    line_img_copy = cv2.cvtColor(line_img.copy(), cv2.COLOR_GRAY2BGR)

                    draw_segments(line_img_copy, [line_segment[i]])
                    draw_classes(line_img_copy, line_segment, classes)

                    key = show_image_and_wait_for_key(line_img_copy, "segment " + str(i))

                    if key == 27:  # ESC
                        break
                    elif key == 123:  # left arrow
                        i -= 1
                    elif key == 32:  # right arrow
                        i += 1
                    else:
                        if key:
                            classes[i] = unichr(key)
                            i += 1
                        else:
                            print("Next character will be uppercase!")
                            key = show_image_and_wait_for_key(line_img_copy, "segment " + str(i))
                            if key in ALLOWED_CHARS:
                                classes[i] = unichr(key).upper()
                                i += 1
                            else:
                                print("Next character will be lowercase!")

                output_classes.extend(classes)
                output_images.extend(images)

            # writing classes to file
            for i in range(len(output_classes)):
                cv2.imwrite(os.path.join(save_path, "{}.png".format(i)), output_images[i])
                f.write("{}.png {}\n".format(i, output_classes[i]))
