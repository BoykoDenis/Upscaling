import numpy as np
import matplotlib.pyplot as plt

from media import media as m
from PIL import Image

path = 'D:\\Datasets\\upscale\\data\\raw\\'
name = 'img'

def load_image(path, name, number):

    img = Image.open(path + name + ' (' + str(number) + ').jpg')

    return img


img = load_image(path, name, 7)
im = np.array(img)
#plt.imshow(im)
#plt.show()
im = m.cut_the_image(im, [240, 240])
im = im.transpose(0, 2, 3, 1)
#print(im)
print(im.shape)
plt.imshow(im[214])
plt.show()