import io
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

import networkx as nx
import matplotlib.pyplot as plt

from text_preprocessing import format_text


# Define the dataset
dataset = pd.read_csv("data.csv")
dataset.drop(columns = dataset.columns[0], inplace=True, axis=1)

# Function to drop all NaN values
dataset = dataset.dropna()

# Dropping the rows which don't have a tag
dataset = dataset[dataset.tag != '-']

# Converting all items in the 'text' column to dtype string
dataset['text'] = dataset['text'].map(str)

# Applying all the necessary preprocessing techniques to our data
for i in range(dataset.shape[0]):
    dataset.iloc[i,1] = format_text(dataset.iloc[i,1])


# Building the GNN Data Class
tags = {'Employers_Payment':0,
          'EmployeesPayment':1,
          'Fund_Name':2,
          'Payments_Type_and_Date':3,
          'Higher_Rate':4,
          'Middle_rate':5,
          'Lower_Rate':6,
          'Number_of_Units':7,
          'Unit_price':8,
          'Fund_Value':9,
          'Split':10
          }

vocabulary = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
labels = [tags[label] for label in dataset['tag']]
texts = []
for string in dataset['text']:
    text = ""
    for tx in string:
        text += tx
    texts.append(text)
max_length = 64
length = len(labels)
num_classes = len(set(labels))


x = []
y = []
label_embeddings = []

for index in range(dataset.shape[0]):
    raw_text = texts[index]
    data = [vocabulary.index(i) for i in list(raw_text) if i in vocabulary]
    if len(data) > max_length:
        data = data[:max_length]
    elif 0 < len(data) < max_length:
        data = data + [0 for _ in range(max_length - len(data))]
    elif len(data) == 0:
        data = [0 for _ in range(max_length)]
    label = labels[index]

    x.append(data)
    y.append(label)

for j in tags:
    data = [vocabulary.index(i) for i in list(j) if i in vocabulary]
    if len(data) > max_length:
        data = data[:max_length]
    elif 0 < len(data) < max_length:
        data = data + [0 for _ in range(max_length - len(data))]
    elif len(data) == 0:
        data = [0 for _ in range(max_length)]

    label_embeddings.append(data)

label_embeddings = label_embeddings[:10]


x = x[:600]
y = y[:600]

x = x*3
y = y*3


classes_x = [[] for _ in range(11)]

for i in range(len(y)):
    classes_x[y[i]].append(x[i])

classes_x = classes_x[:10]

min_length = 1800

for i in range(len(classes_x)):
    min_length = min(min_length, len(classes_x[i]))

for i in range(len(classes_x)):
    classes_x[i] = classes_x[i][:100]


x = []
y = []

for i in range(0, 100, 5):
    for j in range(5):
        x.append(label_embeddings[j])
        y.append(j)

    for k in range(i, i+5):
        for j in range(5):
            x.append(classes_x[j][k])
            y.append(j)

    for j in range(5,10):
        x.append(label_embeddings[j])
        y.append(j)

    for k in range(i, i+5):
        for j in range(5,10):
            x.append(classes_x[j][k])
            y.append(j)


pos = [[] for _ in range(1200)]
for i in range(1200):
    pos[i] = [i//30, (i%30)//5, (i%30)%5]


edge_index_1 = []
edge_index_2 = []

for i in range(0, 1200, 30):
    # Horizontal edges
    for j in range(i, i+26, 5):
        for k in range(j, j+4, 1):
            edge_index_1.append(k)
            edge_index_2.append(k+1)
            edge_index_1.append(k+1)
            edge_index_2.append(k)

    # Vertical edges
    for j in range(i, i+5, 1):
        for k in range(j, j+21, 5):
            edge_index_1.append(k)
            edge_index_2.append(k+5)
            edge_index_1.append(k+5)
            edge_index_2.append(k)

    # Header edges
    for j in range(i, i+5, 1):
        for k in range(j+5, j+26, 5):
            edge_index_1.append(k)
            edge_index_2.append(j)

edge_index = [edge_index_1, edge_index_2]


# Obtaining training, validation and testing masks for our data
from sklearn.model_selection import train_test_split

dataset = pd.DataFrame(x)
dataset['y'] = y

train, test = train_test_split(dataset, test_size=0.1, random_state=42)
train, val = train_test_split(train, test_size=0.1, random_state=42)

def get_mask(index):
    mask = np.repeat([False], 1200)
    mask[index] = True
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask

train_mask = get_mask(train.index)
val_mask = get_mask(val.index)
test_mask = get_mask(test.index)


x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)
pos = torch.tensor(pos, dtype=torch.long)
edge_index = torch.tensor(edge_index, dtype=torch.long)


import torch
from torch_geometric.data import Data

data = Data(x=x, y=y, edge_index=edge_index, pos=pos,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

f = open('graph_properties.txt', 'w+')
f.write('Graph properties \n')
f.write('============================================================== \n')

# Gather some statistics about the graph.
f.write(f'Number of nodes: {data.num_nodes} \n') #Number of nodes in the graph
f.write(f'Number of edges: {data.num_edges} \n') #Number of edges in the graph
f.write(f'Number of features per node: {data.num_node_features} \n')
f.write(f'Average node degree: {data.num_edges / data.num_nodes:.2f} \n') # Average number of nodes in the graph
f.write(f'Contains isolated nodes: {data.has_isolated_nodes()} \n') #Does the graph contains nodes that are not connected
f.write(f'Contains self-loops: {data.has_self_loops()} \n') #Does the graph contains nodes that are linked to themselves
f.write(f'Is undirected: {data.is_undirected()}') #Is the graph an undirected graph

f.close()

# Creating the GNN Model
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(data.num_features, 64)
        self.conv2 = GCNConv(64, 10)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN()


# TSNE Plot to visualize the node embeddings in a 2D space
from visualize_tsne import visualize

model = GCN()
model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y, path='tsne_pre_training.png')


# Training the GNN Model
from training import run_training

epochs = 200
train_acc_list, val_acc_list, test_acc_list = run_training(model, data, epochs=200)


# Plotting the training vs Validation Accuracy

plt.plot(range(epochs), train_acc_list, label='train')
plt.plot(range(epochs), val_acc_list, label='val')
# plt.plot(range(epochs), test_acc_list, label='test')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.savefig('accuracy_plot.png')


model.eval()

out = model(data.x, data.edge_index)
visualize(out, color=data.y, path='tsne_post_training.png')
