import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.optim import AdamW
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report, precision_score
from collections import OrderedDict

import flwr as fl


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
base_path = './ImageData/client_11/'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

def force_three_channel(image):
    if image.mode == 'L':
        # If the image is single-channel (grayscale), convert it to RGB.
        image = image.convert('RGB')
    return image

def load_data():
  
  train_transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.Lambda(force_three_channel),
                                transforms.ToTensor()
  ])
  train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
  print(len(train_dataset))

  test_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.ToTensor()])

  test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
  print(test_dataset.class_to_idx)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

  num_examples = {"trainset" : len(train_dataset), "testset" : len(test_dataset)}

  print(len(train_dataset))

  return train_dataloader, test_dataloader, num_examples

train_dataloader, test_dataloader, num_examples = load_data()


net = models.vgg16(progress=True, pretrained=True)
for param in net.parameters():
  param.requires_grad = False
net.classifier[6] = nn.Linear(in_features=4096, out_features=4)
net = net.to(DEVICE)


from tqdm import tqdm
def train(net, trainloader, epochs: int):
  loss_fn = nn.CrossEntropyLoss().to(DEVICE)
  optimizer = AdamW(net.parameters(),lr = 1e-5)
  for epoch in range(epochs):
    losses= []
    correct_predictions = 0
    for train_image, train_y in tqdm(trainloader):
      train_image, train_y = train_image.to(DEVICE), train_y.to(DEVICE) 
      optimizer.zero_grad()
      outputs = net(train_image)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, train_y)
      correct_predictions += torch.sum(preds == train_y)
      losses.append(loss.item())
      loss.backward()
      optimizer.step()
    print(f"For epoch {epoch} in client - 1 Accuracy: {correct_predictions/num_examples['trainset']}, Loss: {np.mean(losses)}")

def test(net, testloader):
  net.eval()
  loss_fn = nn.CrossEntropyLoss().to(DEVICE)
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for test_image, test_y in tqdm(testloader):
      test_image, test_y = test_image.to(DEVICE), test_y.to(DEVICE) 
      outputs = net(test_image)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, test_y)
      correct_predictions += torch.sum(preds == test_y)
      losses.append(loss.item())
        
  return correct_predictions.double() / num_examples['testset'] , np.mean(losses)

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, train_dataloader, epochs=30)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, loss = test(net, test_dataloader)
        return float(loss), num_examples["testset"], {
           "accuracy": float(acc),
           "loss": float(loss)
           }

fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient(), grpc_max_message_length=1536870912)