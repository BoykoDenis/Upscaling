import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class media():

    def __init__(self, content = None):

        self.__content = content


    def tensor_to_image(self): #tensor -> numpy array

        try:
            self.__content = self.__content.cpu().clone().detach().numpy()
            self.__content = self.__content.squeeze()
            self.__content = self.__content.transpose(1, 2, 0)
            self.__content = self.__content.clip(0, 1)
            return self.__content
        except:
            print("invalid data format")

    def video_to_farmes_iterator(self, filename): # videofile -> tensors

        self.__content = torchvision.io.VideoReader( filename, "video" )
        # do not use yet, wait until update of torchvision

    def get_content(self):

        return self.__content

    def transform_image(transform, data): #for PIL format

        return transform(data)

    def numpy_to_tensor(array):

        return torch.from_numpy(array)

    def cut_the_image(content, into_resolution, meta = False):

        #into_resolution [360, 360]

        # width 2.
        #if content.shape[0] % 2 == 0 and content.shape[1] % 2 == 0:
        if content.shape[0] % into_resolution[0] == 0:
            x = content.shape[0]/into_resolution[0]
            x_padding = 0
        else:
            x = int(content.shape[0]/into_resolution[0])
            x_vborder = (content.shape[0] - (into_resolution[0] * x)) / 2
            x_padding = into_resolution[0] - x_vborder
            x+=2


        #hight 1.

        if content.shape[1] % into_resolution[1] == 0:
            y = content.shape[1]/into_resolution[1]
            y_padding = 0
        else:
            y = int(content.shape[1]/into_resolution[1])
            y_vborder = (content.shape[1] - (into_resolution[1] * y)) / 2
            y_padding = into_resolution[1] - y_vborder
            y+=2

        #swap axis in order to do following parts
        content = content.transpose(2, 0, 1)

        #add padding to allow devision into smaller parts
        content = np.pad(content, (  ( 0, 0 ), ( int(x_padding), int(x_padding) ), ( int(y_padding), int(y_padding) )  )   )

        #swap axis to reshape into batch of 3 chanal size1, size2 parts
        content = np.array(np.split(content, x, axis = 1))
        content = np.array(np.split(content, y, axis = 3))
        content = content.reshape((-1, 3, into_resolution[0], into_resolution[1]))

        meta_data = [x, y, x_padding, y_padding]

        if meta:
            return content, meta_data

        else:
            return content

        #else:
            #return None

    def crop_edges(img, x_edge, y_edge, scale = 2):

        y, x, c = img.shape
        return img[ y_edge*scale:(y - y_edge*scale), x_edge*scale :(x - x_edge*scale), : ]



