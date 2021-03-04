import numpy as np
import matplotlib.pyplot as plt

from media import media as m
from PIL import Image

path = 'D:\\Datasets\\upscale\\data\\raw\\'
save_path = 'D:\\Datasets\\upscale\\data\\ready\\'
name = 'img'

def load_image(path, name, number):

    img = Image.open(path + name + ' (' + str( number ) + ').jpg')

    return img

control_idx = 0
i = 1
while True:
    try:
        img = load_image( path, name, i )
        im = np.array( img )
        im = m.cut_the_image( im, [240, 240] )
        im = im.transpose( 0, 2, 3, 1 )
        for idx, cut in enumerate( im ):

            cut = Image.fromarray(cut)
            cut.save( save_path + name + ' (' + str(int( idx + control_idx + 1 )) + ').jpg' )

            print("loaded: "+ str(int( idx + control_idx + 1 )), end='\r')

        control_idx = idx + control_idx + 1
        i+=1
    except:
        i+=1
        continue