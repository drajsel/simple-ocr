# simple-ocr

A simple approach to optical character recognition (OCR) using OpenCV and k-nearest neighbors algorithm.

## Motivation

I had issues with using Tesseract for recognizing text on a perfect image for OCR (the image was not taken, it was generated). It sometimes mixed up characters Z, 2 and 7, probably because the text wasn't from a natural language, it was a string of alphanumeric characters. I tried training Tesseract for the specific font and adding new training data from my domain, but it didn't help significantly. Since the image is perfectly clean, no noise, I decided to use a simple computer vision approach. Researching existing methods, I stumbled upon a repository mentioning using OpenCV and k-nearest neighbors algorithm (see the credits section) and decided to try it myself. It worked out great for my use case.

## Usage

### Generating training data

I generated my training data by writing all the characters in a text editor in the font I wanted and then screenshooting it.
For example, the image I created for Inconsolata Condensed Regular font:

![alt text](https://github.com/drajsel/simple-ocr/blob/master/data/inconsolata_condensed_regular_train.png)

Using `label.py` the user can label manually the detected characters. 

    usage: label.py [-h] -d DATA -f FONT [-o OUTPUT] [-m {cv2,matplotlib}]
    
    optional arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  Filename of the image containing text for generating
                            training data.
      -f FONT, --font FONT  Name of the font used.
      -o OUTPUT, --output OUTPUT
                            Folder name in the directory 'fonts/[font_name]' in
                            which the data will be stored. For example, you can
                            choose 'train_data' or 'test_data'.
      -m {cv2,matplotlib}, --mode {cv2,matplotlib}
                            Mode of labeling the training data. If the mode is set
                            to 'cv2', then the characters will be labeled by using
                            the keys (the function `cv2.waitKey()`) and the
                            labeling process will be interactively displayed on
                            windows. To enter an uppercase character first press
                            Shift (only 'cv2' mode). Note that this mode doesn't
                            support special characters like č, ć, ... If the mode
                            is set to 'matplotlib', then the characters will be
                            plotted one by one and the user will be prompted to
                            enter the character in the console.
            
The data is saved in `fonts/[font]/[output]` directory.
                            
In the future, I'll try to make this process automatic, by drawing characters using OpenCV in the wanted font (at least for English).

### OCR

The character detection and recognition is performed by the function `classify` in `ocr/classification.py`. One can either use that function or perform the OCR in the command line using `ocr_example.py`.

    usage: ocr_example.py [-h] -i INPUT -f FONT [-s SHOW_STEPS]

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            path to the image on which OCR will be performed
      -f FONT, --font FONT  Name of the font used (as stored in the 'fonts'
                            directory.
      -s SHOW_STEPS, --show_steps SHOW_STEPS
                            Whether or not to display steps of the OCR process.

#### The process:

1. line detection 
    - performing a morphological closing on the image using a rectangle wide in the horizontal direction as a kernel
    - finding contours on the transformed image and storing the bounding boxes

2. character detection
    - performing contour detection on an image of a single line to get the grasp of character dimensions
    - performing a morphological closing on the image using a rectangle wide in the vertical direction as a kernel 
        (the length of the rectangle is a parameter, currently defined as a function of the maximum height and 
        width of the previously detected characters)
    - finding contours on the transformed image and storing the bounding boxes
    
3. character classification
    - performing classification using k-nearest neighbors algorithm trained on annotated data

#### Visualisation of some steps of the process:

original image:

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/hello_world.png)

line detection (morphological closing):

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/line_detection.png)

one of the detected lines:

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/detected_line.png)

character detection (morphological closing):

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/morphing.png)

detected characters (bounding boxes of the detected contours):

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/detected_characters.png)

classification:

![alt text](https://github.com/drajsel/simple-ocr/blob/master/img/ocr_result.png)

## Credits

Work done in this project was inspired by [goncalopp/simple-ocr-opencv](https://github.com/goncalopp/simple-ocr-opencv). The approach in this repository only supports uppercase letters, whereas I needed both lowercase and uppercase, and also letters such as "č", "ć", "š", "ž" and "đ" which aren't supported by the provided labelling (grounding) method. Furthermore, I had to improve the segmentation method which didn't detect multi-part characters, e.g. i, j. I've decided to write my own functions from scratch, not build upon the original code. Nevertheless, I reused some utility functions and the idea of detecting characters from image by finding contours and classifying them with the k-nearest neighbors algorithm.  
