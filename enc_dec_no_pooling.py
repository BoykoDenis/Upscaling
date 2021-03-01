import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from model4 import Enc_Dec

lr = 0.0001

epochs = 10

flag = False

load_m = True

channels_noise = 1800

batch_size = 5

n_featers = 60

input_chanels = 3

DATASET_SIZE = 1000
DATASET_MOVE = 1
DATASET_SCALE = 3


MOD_SAVE_PATH = "models\\model_alpha_second_fina_experiment2.pth.tar"
MOD_LOAD_PATH = "models\\model_alpha_second_fina_experiment2.pth.tar"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def im_convert(tensor):

	image = tensor.cpu().clone().detach().numpy()

	#clone tensor --> detach it from computations --> transform to numpy

	image = image.squeeze()

	#image = image.transpose(1, 2, 0)

	# swap axis from(1,28,28) --> (28,28,1)

	#image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))

	#denormalize image

	image = image.clip(0, 1)

	#sets image range from 0 to 1

	return image*255





def save_mod(state, filename = MOD_SAVE_PATH):

	torch.save(state, filename)

def load_mod(checkpoint):

	model.load_state_dict(checkpoint['state_dictionary'])
	#optimizer.load_state_dict(['state_dictionary'])

transform_train = transforms.Compose([transforms.Resize((360, 640)),
									  transforms.ToTensor(),
									  ])

training_path = 'D:\\Datasets\\Mountain\\mountain\\'

def load_data(DATASET_SIZE = DATASET_SIZE, DATASET_SCALE = DATASET_SCALE, DATASET_MOVE = DATASET_MOVE):

	training_dataset = []

	for i in range(1, DATASET_SIZE):

		img = Image.open(training_path+"train (" + str((i*DATASET_SCALE) + DATASET_MOVE) + ").jpg")
		im = transform_train(img)

		training_dataset.append(im)

		img.close()
		img = None
		del img
		del im
		if i%100 == 0:
			gc.collect()
			print(i, " images have been converted", end = "\r")
	return training_dataset

training_dataset = load_data()

training_dataset = torch.stack(training_dataset)

training_dataset = torch.utils.data.TensorDataset(training_dataset, training_dataset)
training_dataset = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True)

#torch.reshape(training_dataset, (-1, batch_size))
torch.cuda.empty_cache()
#sys.exit()
gc.collect()
criterion = nn.MSELoss(reduction = "sum")
model = Enc_Dec(3, n_featers).to(device)
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr = lr)
graph = []
gc.collect()

if load_m:

	load_mod(torch.load(MOD_LOAD_PATH))

for epoch in range(epochs):
	torch.cuda.empty_cache()
	checkpoint = {"state_dictionary" : model.state_dict(), "optimizer": optimizer.state_dict()}
	RUNNING_LOSS = 0.0


	if epoch % 1 == 0:

		save_mod(checkpoint)

	for idx, [data, label] in enumerate(training_dataset):

		torch.cuda.empty_cache()
		data = data.to(device)
		label = label.to(device)
		model.zero_grad()
		#print(type(label))
		output= model(data)
		#print(output.shape)
		loss = criterion(output, label)
		RUNNING_LOSS += loss.item()

		graph.append(RUNNING_LOSS/(idx+1))

		print("epoch: ", epoch, "dataset progress: ", idx, "loss: ",	RUNNING_LOSS/(idx+1), end = "\r")
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

plt.plot(range(1, len(graph)+1), graph)
plt.show()








