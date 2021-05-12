import PIL.Image
import cv2
import numpy as np
import keras
import extcolors
from utils import cropping_images
from utils import dominant_color


def predict(image, model, color='BGR', height=224, width=224):
    if color == 'BGR':
        im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # conversione in RGB
    else:
        im = image   # se è già in RGB non effettua la conversione

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    pred = model.predict(im)

    mask = pred.reshape((height, width))

    return mask


def transfer(image, mask, threshold):
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    height, width = mask.shape
    array = []
    result = image

    for i in range(height):
        for j in range(width):
            if not mask[i, j]:
                result[i, j, :] = 0
            else:
                array.append(image[i, j, :])
                result[i, j, :] = result[i, j, :]

    # array contiene i colori in formato RGB dei pixel che fanno parte dei capelli
    # result è l'immagine contenente solo la maschera dei capelli nel loro colore originale (RGB)
    return array, result


def hair_color_extractor(image, threshold=0.85):  # immagine in formato BGR
    image_hair = np.uint8(image)
    image_hair_rgb = cv2.cvtColor(image_hair, cv2.COLOR_BGR2RGB)  # conversione in RGB
    # cv2.imshow('image hair', image_hair)
    # cv2.waitKey(0)
    model = keras.models.load_model('hair_model.hdf5')

    mask = predict(image_hair_rgb, model, color='RGB')

    mask, _ = transfer(image_hair, mask, threshold)     # mask è in formato BGR (array)

    frequent_color = dominant_color(mask)
    return frequent_color[2],  frequent_color[1],  frequent_color[0]
