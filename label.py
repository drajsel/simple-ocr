import argparse

from ocr.labeler import label

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data",
                required=True,
                help="Filename of the image containing text for generating training data."
                )
ap.add_argument("-f", "--font",
                required=True,
                help="Name of the font used."
                )
ap.add_argument("-o", "--output",
                required=False,
                default="train_data",
                help="Folder name in the directory 'fonts/[font_name]' in which the data will be stored. "
                     "For example, you can choose 'train_data' or 'test_data'."
                )
ap.add_argument("-m", "--mode",
                required=False,
                default="cv2",
                choices=["cv2", "matplotlib"],
                help="Mode of labeling the training data. If the mode is set to 'cv2', then the characters will be "
                     "labeled by using the keys (the function `cv2.waitKey()`) and the labeling process will be "
                     "interactively displayed on windows. To enter an uppercase character first press Shift"
                     " (only 'cv2' mode). Note that this mode doesn't support special characters like č, ć, ... "
                     "If the mode is set to 'matplotlib', then the characters will be plotted one by one and the user "
                     "will be prompted to enter the character in the console."
                )


args = vars(ap.parse_args())

if __name__ == "__main__":
    label(args['data'], font_name=args['font'], input_mode=args['mode'])
