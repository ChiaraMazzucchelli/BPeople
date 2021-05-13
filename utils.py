import cv2
import math
import os
import face_recognition
import glob
from PIL import Image
import numpy as np
import extcolors
import pandas as pd
import csv
import shutil


def resize(img, scale_percent):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def scaling_images(directory_path, new_directory_path, threshold=300000):
    # directory = os.fsencode(directory_path)
    # new_directory = os.fsencode(new_directory_path)
    for file_path in glob.iglob(directory_path + '/*.*', recursive=False):
        # print('ok')
        file = Image.open(file_path)
        name = os.path.basename(file_path)
        file = np.uint8(file)
        file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
        height, width, _ = file.shape
        area = height * width
        path = new_directory_path + '/' + name

        if area > threshold:
            scale_percent = math.sqrt(threshold / area)
            new_image = resize(file, scale_percent)
            # usare shutil.copy2 per preservare anche i metadati
            # shutil.copy(new_image, new_directory)
            print(path)
            cv2.imwrite(path, new_image)
        else:
            # shutil.copy(file, new_directory)
            cv2.imwrite(path, file)


def cropping_images(image_path, factor=1):
    image = face_recognition.load_image_file(image_path)  # immagine caricata in formato BGR
    top, right, bottom, left = face_recognition.face_locations(image, model="cnn")[0]  # boundind box del viso

    # modifica del boundind box di un fattore dato
    top = int(top / factor)
    right = int(right * factor)
    bottom = int(bottom * factor)
    left = int(left / factor)

    image_cropped = image[0:bottom, left:right]  # crop dell'immagine

    # cv2.imshow('mask', image_cropped)
    # cv2.waitKey(0)

    return image_cropped


def dominant_color(array):
    data = np.reshape(array, (-1, 3))
    # print(data.shape)
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)
    # print(centers)
    print('Dominant color is: RGB({})'.format(centers[0].astype(np.int32)))
    array = centers[0]
    return array
