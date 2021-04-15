import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from media import media as m
from modelUUD import UpscaleUUD as Upscale
from PIL import Image
from PIL import ImageFilter
from PIL.ImageFilter import (GaussianBlur)

#this can be put into a general function
def data_into_tensor_dataloader(training_dataset, labels, batch_size):

    training_dataset = torch.stack(training_dataset)
    labels = torch.stack(labels)
    training_dataset = torch.utils.data.TensorDataset(training_dataset, labels)
    training_dataset = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
    return training_dataset

def load_data(dataset_size, path, name, transform_train, to_tensor):

    f = open("log.txt", "w")
    try:
        labels = []
        data = []
        for i in range(1, dataset_size):
            try:
                loaded_image = Image.open(path + name + ' (' + str( i*100 ) + ').jpg')
                transformed_image = transform_train(loaded_image.convert("RGB").filter(GaussianBlur(radius = 3)))
                label = to_tensor(loaded_image)
                loaded_image.close()

                data.append(transformed_image)
                labels.append(label)
                print("data loaded: ", i, " / ", dataset_size, end = "\r")
            except Exception as ex:
                f.write("loading error: ",str(ex), "\n")
                continue

        return [data, labels]
    except Exception as ex:
        f.write(str(ex)+" -> dataset loading error")
        f.close()

def save_mod(state, path, name):

	torch.save(state, path+name)


#init
model_save_path = 'models/'
model_name = 'model_tUUDk11x5.pth.tar'
path = 'data/ready/'
name = 'img'
dataset_size = 6000
batch_size = 3
epochs = 50
lr = 0.0001

device = torch.device("cuda")

model = Upscale(3, 3).to(device)
parameters = model.parameters()
optimizer = torch.optim.Adam( parameters, lr = lr )
criterion = nn.MSELoss(reduction = "sum")




transform_train = transforms.Compose([transforms.Resize((120, 120)),
                                      transforms.ToTensor(),
                                     ])
to_tensor = transforms.ToTensor()

#data load
training_dataset, labels = load_data(dataset_size, path, name, transform_train, to_tensor)
data_loader = data_into_tensor_dataloader(training_dataset, labels, batch_size)
f = open("log.txt", "w")

try:
    for epoch in range(epochs):

        running_loss = 0.0
        checkpoint = {"state_dictionary" : model.state_dict(), "optimizer": optimizer.state_dict()}
        save_mod(checkpoint, model_save_path, model_name)

        for idx, [data, label] in enumerate(data_loader):

            data = data.to(device)
            label = label.to(device)

            out = model(data)
            loss = criterion(out, label)

            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            f.write(str(["epoch: ", epoch, "dataset progress: ", idx, "loss: ",	running_loss/(idx+1)])+"\n")
            #print("epoch: ", epoch, "dataset progress: ", idx, "loss: ",	running_loss/(idx+1), end = "\r")
    f.close()

except Exception as ex:
    f.write(str(ex)+" -> network training error")
    f.close()