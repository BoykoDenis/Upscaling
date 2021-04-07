import cv2
import torch
import model
import numpy as np

import matplotlib.pyplot as plt
from media import media as m
from PIL import Image

path = 'data\\out (356)croped.jpg'
save_path = 'data\\out (356).jpg'

def upscale( path ):

    def load_image( path ):

        img = Image.open(path)
        return img

    def cut( path ):
        try:
            img = load_image( path )
            im = np.array( img )
            im, meta = m.cut_the_image( im, [240, 240], meta = True )
            #im = im.transpose( 0, 2, 3, 1 )
            return [im, meta]

        except Exception as ex:
            print(ex)

    def upscaler( input_image ):

        device = torch.device("cuda")
        model_save_path = 'models/'
        model_name = 'model_t.pth.tar'
        mod = model.Upscale(3, 3).to(device)
        checkpoint = torch.load(model_save_path + model_name)
        mod.load_state_dict(checkpoint["state_dictionary"])
        input_image = torch.FloatTensor(input_image).to(device)

        if len(input_image.shape) == 4:
            output_image = []
            for image in input_image:
                out = mod(image.unsqueeze(0))
                output_image.append(out)

            output_image = torch.cat(output_image)
            output_image = output_image.cpu().clone().detach().numpy()#.transpose(0, 2, 3, 1)
            return output_image

        elif len(input_image.shape) == 3:
            im = im.unsqueeze(0)
            out = mod(im)
            out = out.cpu().clone().detach().numpy()
            return out

    def recover( im, meta ):

        #np.concatenate()
        im = im.reshape(meta[1], meta[0], 3, 480, 480)
        output_image_main = []
        for column in im:
            if len(output_image_main):
                output_image_part = np.concatenate(column, axis = 1)
                output_image_main = np.concatenate((output_image_main, output_image_part), axis = 2)

            else:
                output_image_part = np.concatenate(column, axis = 1)
                output_image_main = output_image_part
        output_image_main = output_image_main.transpose(1, 2, 0)
        output_image_main = m.crop_edges(output_image_main, int(meta[2]), int(meta[3]))
        return output_image_main

    cut_images, meta = cut( path )
    image = upscaler(cut_images)
    output_image = recover(image, meta)

    return output_image

img = upscale( path )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(save_path, img)

plt.imshow(img/255)
plt.show()





