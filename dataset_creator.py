import numpy as np
import matplotlib.pyplot as plt

from media import media as m
from PIL import Image

path = 'data\\raw\\img'
save_path = 'data\\ready\\img'

def load_image(path, number):

    img = Image.open(path + ' (' + str( number ) + ').jpg')

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
            cut.save( save_path + ' (' + str(int( idx + control_idx + 1 )) + ').jpg' )

            print("loaded: "+ str(int( idx + control_idx + 1 )), end='\r')

        control_idx = idx + control_idx + 1
        i+=1
    except:
        i+=1
        continue