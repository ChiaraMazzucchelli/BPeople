import cv2
import numpy as np


# segment using otsu binarization and thresholding
def segment_otsu(image_grayscale, image):
    threshold_value, threshold_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold_image_binary = 1 - threshold_image / 255
    threshold_image_binary = np.repeat(threshold_image_binary[:, :, np.newaxis], 3, axis=2)
    img_face_only = np.multiply(threshold_image_binary, image)
    return img_face_only


def skin_color_extractor(image, color='BGR'):
    image = np.uint8(image)
    # if not color == 'BGR':
    #    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if color == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imageRGB = image    # immagine in formato RGB
    img_grayscale = cv2.cvtColor(imageRGB, cv2.COLOR_BGR2GRAY)  # conversione in scala di grigi

    # foreground and background segmentation (otsu)
    img_face_only = segment_otsu(img_grayscale, imageRGB)   # segmentazione del viso con la tecnica otsu

    # convert to HSV and YCrCb color spaces and detect potential pixels
    img_face_only_uint8 = np.uint8(img_face_only)
    img_HSV = cv2.cvtColor(img_face_only_uint8, cv2.COLOR_BGR2HSV)      # conversione nello spazio di colore HSV
    img_YCrCb = cv2.cvtColor(img_face_only_uint8, cv2.COLOR_BGR2YCrCb)  # conversione nello spazio di colore YCrCb

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    height, width, channels = img_face_only_uint8.shape

    for i in range(height):
        for j in range(width):
            if ((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (
                    90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img_face_only_uint8[i, j].item(0))
                green.append(img_face_only_uint8[i, j].item(1))
                red.append(img_face_only_uint8[i, j].item(2))

            else:
                img_face_only_uint8[i, j] = [0, 0, 0]

    # determine mean skin tone estimate
    skin_tone_estimate_RGB = [np.mean(red), np.mean(green), np.mean(blue)]
    print("mean skin tone estimate (RGB)", skin_tone_estimate_RGB[0], skin_tone_estimate_RGB[1],
          skin_tone_estimate_RGB[2])
    return skin_tone_estimate_RGB[0], skin_tone_estimate_RGB[1], skin_tone_estimate_RGB[2]
