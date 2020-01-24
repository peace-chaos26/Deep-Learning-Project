import pandas as pd
import numpy as np
import cv2

width, height = 48, 48

data = pd.read_csv('./data/fer2013/fer2013.csv')

pixels = data['pixels'].tolist()
faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    face = np.asarray(face).reshape(width, height)
    # face = face / 255.0
    face = cv2.resize(face.astype('uint8'), (width, height))
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

np.save('./data/faces_array.npy', faces)
