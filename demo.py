import os
import argparse

import pandas as pd
import cv2
from utils import cropping_images
from skin_color_extractor import skin_color_extractor
from hair_color_extractor import hair_color_extractor
from eye_color_extractor import eye_color_extractor


def extract_color_skin_hair(img_path):
    image = cropping_images(img_path, factor=1.3)
    rs, gs, bs = skin_color_extractor(image, color='BGR')
    rh, gh, bh = hair_color_extractor(image)
    return rs, gs, bs, rh, gh, bh


def extract_color_eye(img_path):
    img = cv2.imread(img_path)  # immagine in formato RGB
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = eye_color_extractor(img)
    return r, g, b


def extract_colors(image_path):

    skin_r, skin_g, skin_b, hair_r, hair_g, hair_b = extract_color_skin_hair(image_path)
    eye_r, eye_g, eye_b = extract_color_eye(image_path)
    print("Colore pelle:  ( " + skin_r + ", " + skin_g + ", " + skin_b + " )")
    print("Colore capelli:  ("+ hair_r + ", " + hair_g + ", " + hair_b + " )")
    print("Colore occhi:  (" + eye_r + ", " + eye_g + ", " + eye_b + " )")


parser = argparse.ArgumentParser()

parser.add_argument('--image_path', help="path della cartella da dove prendere le immagini")
# parser.add_argument('--origin_path', help="path del csv di origine")
# parser.add_argument('--new_path', help="path di destinazione del nuovo csv")
# parser.add_argument('--new_name', help="nome del nuovo csv")
args = parser.parse_args()

extract_colors(args.image_path)
