import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from modelUUD import UpscaleUUD as model
from media import media as m
from PIL import Image

path = 'data\\dalibor1.jpg'
save_path = 'data\\dalibor3out21.jpg'

def upscale( path ):

    def load_image( path ):

        img = Image.open(path)#.convert("L")
        plt.imshow(img)
        plt.show()
        return img

    def cut( path, chanels = 3 ):
        try:
            img = load_image( path )
            im = np.array( img )
            im, meta = m.cut_the_image( im, [240, 240], meta = True, chanels=chanels )
            #im = im.transpose( 0, 2, 3, 1 )
            return im, meta

        except Exception as ex:
            print(ex)
            #print('here')

    def upscaler( input_image ):

        with torch.no_grad():

            torch.cuda.empty_cache()

            device = torch.device("cuda")
            model_save_path = 'models/'
            model_name = 'model_tUUDk21p19x4.pth.tar'
            mod = model(3, 3).to(device)

            torch.cuda.empty_cache()

            checkpoint = torch.load(model_save_path + model_name)
            mod.load_state_dict(checkpoint["state_dictionary"])
            input_image = torch.FloatTensor(input_image)#.to(device)

            torch.cuda.empty_cache()

            if len(input_image.shape) == 4:
                output_image = []
                for image in input_image:
                    out = mod(image.unsqueeze(0).to(device)).to("cpu")
                    output_image.append(out)
                    torch.cuda.empty_cache()

                output_image = torch.cat(output_image)
                output_image = output_image.cpu().clone().detach().numpy()
                return output_image

            elif len(input_image.shape) == 3:
                im = im.unsqueeze(0)
                out = mod(im)
                out = out.cpu().clone().detach().numpy()
                torch.cuda.empty_cache()
                return out

    def recover( im, meta, chanels = 3 ):

        torch.cuda.empty_cache()

        im = im.reshape(int(meta[1]), int(meta[0]), int(chanels), 480, 480)
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

    cut_images, meta = cut( path, chanels = 3 )
    image = upscaler(cut_images)
    output_image = recover(image, meta, chanels = 3)

    return output_image

img = upscale( path )
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(save_path, img)

plt.imshow(img/255)
plt.show()





