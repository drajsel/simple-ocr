import cv2
import numpy as np
import os
import shutil
import time

from ocr.segmentation import process_image, region_from_segment
from ocr.processing import autocrop, remove_noise, add_padding_to_characters

from sklearn.neighbors import KNeighborsClassifier
from skimage.metrics import structural_similarity
from scipy.spatial.distance import pdist, squareform

IMG_EXTENSIONS = ["jpg", "jpeg", "png"]
BLUR_SIZE = 1


def image_euclidean_distance(x, y, kernel=None):
    """
    A generalized way to calculate the Euclidean distance for matrices,
    more suitable for images.
    See paper "On the Euclidean Distance of Images" by Wang et al. for details.
    The kernel is a distance matrix for the matrix indices.
    Since it's calculated only once , we calculate it
    before the classification algorithm initialization.

    :param x, y: flattened images whose distance is to be calculated using this metric
    :param kernel: kernel applied in the distance calculation
    :return dist: calculated distance between flattened images x and y
    """

    diff = np.array(x - y).reshape(len(x), 1)
    dist = float(np.matmul(np.transpose(diff), np.matmul(kernel, diff)).squeeze())

    return dist


def process_training_data(font_name, label_file="labels.txt", folder_name="train_data",
                          feature_size=50, new_folder_path=None,
                          padding_eps=None):
    """
    The function loads the image data from folder_path and processes the images
    (grayscale -> blur -> thresholding -> optional padding -> resize to FEATURE_SIZExFEATURE_SIZE )
    and saves the processed images to new_folder_path.

    :param font_name: name of the font
    :param label_file: name of the file containing image filenames and classes
    :param folder_name: path of the folder which contains text file "labels.txt" containing image
    :param feature_size: size of the resulting image (in each dimension)
    :param new_folder_path: path of the folder to which the images will be saved
    :param padding_eps: if padding_eps is not None, padding_eps is the threshold used to determine
        which characters should be padded
    """

    folder_path = os.path.join("data", "fonts", font_name, folder_name)
    img_paths = os.listdir(folder_path)
    img_paths = [img_path for img_path in img_paths if img_path.split(".")[-1] in IMG_EXTENSIONS]

    if new_folder_path is None:
        new_folder_path = os.path.join("data", "fonts", font_name + "_{0}x{0}".format(feature_size), folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

    shutil.copy(os.path.join(folder_path, label_file), os.path.join(new_folder_path, label_file))

    for img_path in img_paths:
        img = cv2.imread(os.path.join(folder_path, img_path), cv2.IMREAD_GRAYSCALE)

        # cv2.imshow("original character", img)
        # cv2.waitKey(0)

        # TODO: blur or not to blur? if the blur size is equal to 1,
        img = cv2.blur(img, (BLUR_SIZE, BLUR_SIZE), 0)
        # blurred = cv2.GaussianBlur(img,(BLUR_SIZE,BLUR_SIZE),cv2.BORDER_DEFAULT)

        # img_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 6)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1] # THRESH_OTSU

        if padding_eps is not None:
            img = add_padding_to_characters(img, eps=padding_eps)

        img = cv2.resize(img, (feature_size, feature_size))

        # cv2.imshow("processed character", img)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join(new_folder_path, img_path), img)


def load_data_from_folder(folder_path, label_file="labels.txt"):
    """
    Read training/test data from specified folder and extract features from images,
    i.e. resize the image to FEATURE_SIZExFEATURE_SIZE and flatten the matrix.
    The features are saved to 'features' and the character classes are mapped to integers. The mapping
    from integer to character classes is stored in `number_to_class` dictionary.

    :param folder_path: path of the folder containing labelled images and the text document containing the labels
                            (each line contains image filename and class name in that order, separated by space)
    :param label_file: name of the text document containing the labels

    :return features: list containing vector representations of the images
    :return int_classes: list containing classes mapped to integers
    :return int_to_class: dictionary mapping integers to true class names

    """

    with open(os.path.join(folder_path, label_file), "r") as f:
        lines = f.readlines()

    features = []
    classes = []
    int_to_class = {}

    for line in lines:
        line = line.strip("\n").strip(" ")
        img_path, char_class = line.split(" ")

        number = int(img_path.split(".")[0])
        int_to_class.update({number: char_class})

        # read the character image as gray and reshape into a vector
        img = cv2.imread(os.path.join(folder_path, img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]

        # cv2.imshow("train char", img)
        # cv2.waitKey(0)

        features.append(img.flatten())
        classes.append(char_class)

    int_classes = list(int_to_class.keys())
    features = np.array(features)

    return features, int_classes, int_to_class


def prepare_knn_classifier(train_data_path, metric='euclidean'):
    """
    Function that "trains" the k-nearest neighbour classifier and returns the model.

    :param train_data_path: path to the folder containing training data (images and text file containing labels)
    :param metric: distance metric to be used in the knn algorithm

    :return model: trained k-nn classifier
    :return train_dict: dictionary mapping integers to class labels
    :return char_to_dict: dictionary mapping class labels to integers

    """

    train_features, train_classes, train_dict = load_data_from_folder(train_data_path)
    char_to_int = {train_dict[key]: key for key in train_dict}

    feature_size = int(np.sqrt(len(train_features[0])))

    if metric.lower() == 'imed':
        pixel_inds = np.arange(feature_size**2).reshape(feature_size**2, 1)
        kernel = np.exp(-squareform(pdist(pixel_inds, 'euclidean'))**2 / 2)
        model = KNeighborsClassifier(
            n_neighbors=1,
            metric=lambda x,y: image_euclidean_distance(x, y, kernel=kernel)
        )
    elif metric.lower() == 'similarity_index':
        model = KNeighborsClassifier(
            n_neighbors=1,
            metric=lambda x, y: 1 - (structural_similarity(
                x.reshape(feature_size, feature_size),
                y.reshape(feature_size, feature_size),
                data_range=255) + 1) / 2
        )
    else:
        # 'euclidean' metric
        model = KNeighborsClassifier(n_neighbors=1)

    model.fit(train_features, train_classes)

    return model, train_dict, char_to_int


def classify(img, font_name="inconsolata_condensed_regular", feature_size=50, show_steps=False, metric='imed'):
    """
    Function performing the classification on an image containing text of the same font.

    :param img: image containing text
    :param font_name: font name
    :param feature_size: if the training data is not yet processed,
                            the processing will be performed with the particular feature size
    :param show_steps: if True, the steps of the process will be visualised
    :param metric: distance metric used in the classification algorithm
    :return ocr_result: result of the knn classification on the detected text in image
    """

    PAD = 20
    # sometimes the contours are off by one/several pixels so the OFFSET variables corrects the boxes
    OFFSET_Y = 0
    OFFSET_X = 0

    if "condensed" in font_name:
        padding_eps = 0.6
    else:
        padding_eps = 0.3

    img = remove_noise(img)
    # pad image so the contours are found properly
    img = cv2.copyMakeBorder(img, top=PAD, bottom=PAD, left=PAD, right=PAD, borderType=cv2.BORDER_CONSTANT,
                             value=(255, 255, 255))

    # process training data if it hasn't been done already
    train_data_path = os.path.join("data", "fonts", font_name + "_{0}x{0}".format(feature_size), "train_data")
    if not os.path.exists(train_data_path):
        process_training_data(font_name, feature_size=feature_size, padding_eps=padding_eps)

    # prepare the classifier
    model, train_dict, char_to_int = prepare_knn_classifier(train_data_path, metric=metric)

    # process the image; break it up into lines and then each line into characters
    lines, line_imgs = process_image(img, show_steps=show_steps, offset_x=OFFSET_X, offset_y=OFFSET_Y)

    # extract and classify characters
    ocr_result = []
    for line, line_img in zip(lines, line_imgs):
        max_width = np.max(line[:,2])
        line_img = np.uint8(line_img)

        # detecting space between characters based on the bounding box coordinates
        space_inds = []
        for i in range(len(line)-1):
            x_l, _, w_l, _ = line[i]
            x_r, _, w_r, _ = line[i+1]

            # it the distance between two adjacent bounding boxes is larger than some threshold, assume
            # that that is a space character (does not recognize multiple spaces currently)
            # TODO: check the threshold max_width for edge cases
            if x_r > max_width + (x_l + w_l):
                space_inds.append(i+1)

        # feature extraction
        features = []
        for segment in line:
            box = np.uint8(region_from_segment(line_img, segment))
            box = autocrop(box)
            box = cv2.erode(box, (3,3))
            box = cv2.blur(box, (BLUR_SIZE, BLUR_SIZE), 0)
            # box = cv2.threshold(box, 0, 255, cv2.THRESH_OTSU)[1]
            box = add_padding_to_characters(box, eps=padding_eps)
            box = cv2.resize(box, (feature_size, feature_size))
            box = cv2.threshold(box, 0, 255, cv2.THRESH_OTSU)[1]

            # cv2.imshow("char", box)
            # cv2.waitKey(0)

            features.append(box.flatten())

        # classify detected characters

        start_class = time.time()
        predictions = model.predict(features)
        # print("Classifying time: ", int(1000*(time.time() - start_class)))
        classes = [train_dict[pred] for pred in predictions]

        if show_steps:
            for i in range(len(classes)):
                x, y, w, h = line[i]

                display_rect(line_img, line[i])
                cv2.putText(line_img, classes[i], (x,y),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,0,0), thickness=2)
            cv2.imshow("line OCR result", line_img)
            cv2.waitKey(0)

        # insert detected spaces
        for i, space_ind in enumerate(space_inds):
            classes.insert(space_ind + i, " ")

        ocr_result.append("".join(classes))

    ocr_result = "\n".join(ocr_result)

    return ocr_result
