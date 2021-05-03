import cv2
import keras
# from tensorflow.python.client import device_lib
import pandas as pd
import os

from hair_color_extractor import hair_color_extractor, predict, transfer
from skin_color_extractor import skin_color_extractor, segment_otsu
from eye_color_extractor import eye_color_extractor
from utils import resize, scaling_images, cropping_images
import tensorflow as tf
from create_csv import extract_all_colors, extract_color_eye, extract_color_skin_hair
from PIL import Image
import numpy as np
import random


def resize_image():
    images_path = 'Images'
    resized_images_path = 'ResizedImages'
    scaling_images(images_path, resized_images_path)


if __name__ == '__main__':
    image_path = 'ResizedImages/10.jpg'
    images_path = 'ResizedImages'
    csv_path = 'dataset_new.csv'
    new_csv_path = 'C:\\Users\\Huawei\\PycharmProjects\\pythonProject\\datasets'
    new_csv_name = 'dataset_completo'

    # extract_all_colors(images_path, csv_path, new_csv_path, new_csv_name)
    # _, _, _, r, g, b = extract_color_skin_hair(image_path)
    # print(r, g, b)
