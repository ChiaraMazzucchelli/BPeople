import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from utils import dominant_color


def eye_color_extractor(image):
    detector = MTCNN()
    h, w = image.shape[0:2]
    img_mask = np.zeros((image.shape[0], image.shape[1], 1))

    result = detector.detect_faces(image)
    if not result:
        print('Warning: Can not detect any face in the input image!')
        return

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = eye_distance / 15  # approximate

    img_mask = cv2.circle(img_mask, left_eye, int(eye_radius), (255, 255, 255), -1)
    img_mask = cv2.circle(img_mask, right_eye, int(eye_radius), (255, 255, 255), -1)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    array = []
    for y in range(0, h):
        for x in range(0, w):
            if img_mask[y, x] == 0:
                image[y, x, :] = 0
            else:
                array.append(image[y, x, :])

    array = dominant_color(image)
    return array[2], array[1], array[0]