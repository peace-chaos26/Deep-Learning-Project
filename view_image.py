import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

width, height = 48, 48

data = pd.read_csv('./data/fer2013/fer2013.csv')

pixels = data['pixels'].tolist()
num = int(input("Enter a number: "))

face = [int(pixel) for pixel in pixels[num].split(' ')]
face = np.asarray(face).reshape(width, height)
face = cv2.resize(face.astype('uint8'), (width, height))

print(data['emotion'][num])

imgplot = plt.imshow(face, cmap='gray')
plt.show()
