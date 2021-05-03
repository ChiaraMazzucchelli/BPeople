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
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = eye_color_extractor(img)
    return r, g, b


def extract_all_colors(dataset_path, csv_path, new_csv_path, new_csv_name):
    skin_red = []
    skin_green = []
    skin_blue = []
    hair_red = []
    hair_green = []
    hair_blue = []
    eye_red = []
    eye_green = []
    eye_blue = []
    dataset = pd.read_csv(csv_path)

    # for path in os.listdir(dataset_path):
    for row in dataset.index:
        name = dataset._get_value(row, 'ID')
        ext = dataset._get_value(row, 'Extension')
        path = str(name) + '.' + ext
        print('Image = ' + path)
        if name in {26, 37, 49, 68, 74, 87, 94, 95, 103, 108}:
            skin_red.append(0)
            skin_green.append(0)
            skin_blue.append(0)
            hair_red.append(0)
            hair_green.append(0)
            hair_blue.append(0)
            eye_red.append(0)
            eye_green.append(0)
            eye_blue.append(0)
        else:
            full_path = os.path.join(dataset_path, path)
            skin_r, skin_g, skin_b, hair_r, hair_g, hair_b = extract_color_skin_hair(full_path)
            skin_red.append(skin_r)
            skin_green.append(skin_g)
            skin_blue.append(skin_b)
            hair_red.append(hair_r)
            hair_green.append(hair_g)
            hair_blue.append(hair_b)
            eye_r, eye_g, eye_b = extract_color_eye(full_path)
            eye_red.append(eye_r)
            eye_green.append(eye_g)
            eye_blue.append(eye_b)

    dataset["Skin_R"] = skin_red
    dataset["Skin_G"] = skin_green
    dataset["Skin_B"] = skin_blue
    dataset["Hair_R"] = hair_red
    dataset["Hair_G"] = hair_green
    dataset["Hair_B"] = hair_blue
    dataset["Eye_R"] = eye_red
    dataset["Eye_G"] = eye_green
    dataset["Eye_B"] = eye_blue

    path = new_csv_path + '\\' + new_csv_name + '.csv'
    dataset.to_csv(path, index=False)
    print('Dataset creato correttamente nella directory ' + path)


parser = argparse.ArgumentParser()

parser.add_argument('--images_path', help="path della cartella da dove prendere le immagini")
parser.add_argument('--origin_path', help="path del csv di origine")
parser.add_argument('--new_path', help="path di destinazione del nuovo csv")
parser.add_argument('--new_name', help="nome del nuovo csv")
args = parser.parse_args()

extract_all_colors(args.images_path, args.origin_path, args.new_path, args.new_name)
