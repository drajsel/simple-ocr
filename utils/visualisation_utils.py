import cv2


def ask_for_key(return_arrow_keys=True):
    key = 128
    while key > 127:
        key = cv2.waitKey(0)

        if return_arrow_keys:
            if key in (65362, 65364, 65361, 65363):  # up, down, left, right
                return key
        key %= 256
    return key


def show_image_and_wait_for_key(image, name="Image"):
    """
    Shows an image, outputting name. keygroups is a dictionary of keycodes to functions;
    they are executed when the corresponding keycode is pressed
    """

    print("showing", name, "(waiting for input)")
    cv2.imshow('norm', image)
    return ask_for_key()


def draw_segments(image, segments, color=(255, 0, 0), line_width=1):
    """draws segments on image"""
    for segment in segments:
        x, y, w, h = segment
        cv2.rectangle(image, (x, y), (x + w, y + h), color, line_width)


def draw_lines(image, ys, color=(255, 0, 0), line_width=1):
    """draws horizontal lines"""
    for y in ys:
        cv2.line(image, (0, y), (image.shape[1], y), color, line_width)


def draw_classes(image, segments, classes):
    assert len(segments) == len(classes)
    for s, c in zip(segments, classes):
        x, y, w, h = s
        cv2.putText(image, c, (x, y), 0, 0.5, (128, 128, 128))