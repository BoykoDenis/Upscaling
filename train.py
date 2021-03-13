import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from media import media as m
from model import Upscale
from PIL import Image

#this can be put into a general function
def data_into_tensor_dataloader(training_dataset, labels, batch_size):
    training_dataset = torch.stack(training_dataset)
    labels = torch.stack(labels)
    training_dataset = torch.utils.data.TensorDataset(training_dataset, labels)
    training_dataset = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
    return training_dataset

def load_data(dataset_size, path, name, transform_train, to_tensor):

    labels = []
    data = []
    for i in range(1, dataset_size):

        loaded_image = Image.open(path + name + ' (' + str( i ) + ').jpg')
        transformed_image = transform_train(loaded_image)
        label = to_tensor(loaded_image)
        loaded_image.close()

        data.append(transformed_image)
        labels.append(label)

    return [data, labels]

def save_mod(state, path, name):

	torch.save(state, path+name)


#init
model_save_path = 'models\\'
model_name = 'model_t'
path = 'D:\\Datasets\\upscale\\data\\ready\\'
name = 'img'
dataset_size = 5 #776486
batch_size = 5
epochs = 1
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Upscale(3, 3).to(device)
parameters = model.parameters()
optimizer = torch.optim.SGD( parameters, lr = lr )
criterion = nn.MSELoss(reduction = 'sum')




transform_train = transforms.Compose([transforms.Resize((120, 120)),
                                      transforms.ToTensor(),
                                     ])
to_tensor = transforms.ToTensor()

#data load
training_dataset, labels = load_data(dataset_size, path, name, transform_train, to_tensor)
data_loader = data_into_tensor_dataloader(training_dataset, labels, batch_size)


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

        print("epoch: ", epoch, "dataset progress: ", idx, "loss: ",	running_loss/(idx+1), end = "\r")
