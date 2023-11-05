from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report, precision_score
import numpy as np
import flwr as fl

df = pd.read_csv('./TextData/text_second_client.csv')
df['description'] = df['description'].apply(lambda x: x[0:500])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

max_description_len = train_df['description'].str.split().map(lambda x: len(x)).max()
extra_tokens = 20

train_text = train_df['description']
train_labels = train_df['label']

test_text = test_df['description']
test_labels = test_df['label']

bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    padding=True,
    truncation=True,
    max_length=max_description_len + extra_tokens
)


# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    padding=True,
    truncation=True,
    max_length=max_description_len + extra_tokens
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.values.tolist())


test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.values.tolist())

def load_data():
  BATCH_SIZE = 2
  train_data = TensorDataset(train_seq, train_mask, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)

  test_data = TensorDataset(test_seq, test_mask, test_y)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE, drop_last=True)

  num_examples = {"trainset" : len(train_data), "testset" : len(test_data)}

  return train_dataloader, test_dataloader, num_examples

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(net.parameters(),lr = 1e-5)
    net.train()
    
    for epoch in range(epochs):
        losses= []
        correct_predictions = 0
        for train_seq, train_mask, train_y in trainloader:
          train_seq, train_mask, train_y = train_seq.to(device), train_mask.to(device), train_y.to(device)
          optimizer.zero_grad()
          outputs = net(train_seq, train_mask)
          _, preds = torch.max(outputs, dim=1)

          loss = loss_fn(outputs, train_y)
          correct_predictions += torch.sum(preds == train_y)
          losses.append(loss.item())
          loss.backward()
          optimizer.step()
        print(f"For epoch 1 in client - 2 Accuracy: {correct_predictions/len(train_df)}, Loss: {np.mean(losses)}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    net.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    losses = []
    correct_predictions = 0
    with torch.no_grad():
      for test_seq, test_mask, test_y in testloader:
        test_seq, test_mask, test_y = test_seq.to(device), test_mask.to(device), test_y.to(device)
        outputs = net(test_seq, test_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, test_y)
        correct_predictions += torch.sum(preds == test_y)
        losses.append(loss.item())
        
    return correct_predictions.double() / len(test_df) , np.mean(losses)


class Net(nn.Module): #1 D CNN MOdel
    def __init__(self, bert):
        super(Net, self).__init__()
        self.bert = bert
        self.conv = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding='valid', stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.clf1 = nn.Linear(256, 256)
        self.clf2 = nn.Linear(256, 4)

    # Define the forward pass
    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = cls_hs[0]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)

        # Apply adaptive max pooling along the sequence_length dimension
        x = nn.functional.adaptive_max_pool1d(x, 1)  # Pooling with kernel_size=1

        x = x.view(x.size(0), -1)  # Flatten the tensor along the spatial dimensions
        x = self.clf1(x)
        
        return self.clf2(x)

# Load model and data
net = Net(bert).to(device)
trainloader, testloader, num_examples = load_data()

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc, loss = test(net, testloader)
        return float(loss), num_examples["testset"], {
           "accuracy": float(acc),
           "loss": float(loss)
           }


fl.client.start_numpy_client(server_address="localhost:8080", client=CifarClient(), grpc_max_message_length=1536870912)