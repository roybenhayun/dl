# -*- coding: utf-8 -*-
"""22961_6_3_RNN_classifiers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Idan-Alter/OU-22961-Deep-Learning/blob/main/22961_6_3_RNN_classifiers.ipynb
"""



import torch
from torch import nn
import datasets as ds
from pprint import pprint
from tqdm import tqdm
from matplotlib import pyplot as plt


dataset = ds.load_dataset("glue", "sst2")

sentence_list = dataset["train"]["sentence"]
labels_list   = dataset["train"]["label"]
tokenize      = lambda x: x.split()
tokenized     = list(map(tokenize, sentence_list))

from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(tokenized, specials=["<UNK>"], min_freq=5)
vocab.set_default_index(0)

func = lambda x: torch.tensor(vocab(x))
integer_tokens = list(map(func, tokenized))
label_tensors  = list(map(torch.tensor, labels_list))
print(*sentence_list[1:3], sep="\n")
print(*integer_tokens[1:3], sep="\n")
print(*label_tensors[1:3], sep="\n")

test_split   = len(integer_tokens) * 8//10
train_tokens = integer_tokens[:test_split]
train_labels = label_tensors[:test_split]
test_tokens  = integer_tokens[test_split:]
test_labels  = label_tensors[test_split:]

tanh = nn.Tanh()
x    = torch.arange(-2,2,0.1)
y    = tanh(x)
plt.plot(x,y);
plt.show()

class MyRNNCell(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_state = torch.zeros(hidden_dim)
        self.hidden_linear= nn.Linear(in_features = hidden_dim,
                                     out_features = hidden_dim)
        self.input_linear = nn.Linear(in_features = embed_dim,
                                     out_features = hidden_dim)
        self.activation   = nn.Tanh()
    def forward(self, one_embedded_token):
        Z1        = self.input_linear(one_embedded_token)
        Z2        = self.hidden_linear(self.hidden_state)
        Y         = Z1+Z2
        new_state = self.activation(Y)
        self.hidden_state = new_state
        #return

class RNNClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding  = nn.Embedding(len(vocab),embed_dim)
        self.rnn        = MyRNNCell(embed_dim, hidden_dim)
        self.linear     = nn.Linear(hidden_dim, 2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, sentence_tokens):
      self.rnn.hidden_state = torch.zeros(self.hidden_dim)
      for one_token in sentence_tokens:
        one_embedded_token = self.embedding(one_token)
        self.rnn(one_embedded_token)

      feature_extractor_output = self.rnn.hidden_state
      class_scores       = self.linear(feature_extractor_output)
      logprobs           = self.logsoftmax(class_scores)
      return logprobs

class DeepRNNClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, RNNlayers=2):
        super().__init__()
        assert RNNlayers >= 1
        self.hidden_dim = hidden_dim
        self.embedding  = nn.Embedding(len(vocab), embed_dim)
        # self.rnn1    = MyRNNCell(embed_dim, hidden_dim)
        # self.rnn2    = MyRNNCell(hidden_dim, hidden_dim)      #
        self.linear     = nn.Linear(hidden_dim, 2)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        # change #1
        self.rnn_list = []
        self.rnn_list.append(MyRNNCell(embed_dim, hidden_dim))
        for i in range(RNNlayers - 1):
            self.rnn_list.append(MyRNNCell(hidden_dim, hidden_dim))

    def forward(self, sentence_tokens):
      for rnn_cell in self.rnn_list:
          rnn_cell.hidden_state = torch.zeros(self.hidden_dim)
      for one_token in sentence_tokens:
        one_embedded_token = self.embedding(one_token)
        self.rnn_list[0](one_embedded_token)
        prev_hidden_state = self.rnn_list[0].hidden_state
        for rnn_cell in self.rnn_list[1:]:
            rnn_cell(prev_hidden_state)
            prev_hidden_state = rnn_cell.hidden_state

      feature_extractor_output = prev_hidden_state       #
      class_scores     = self.linear(feature_extractor_output)
      logprobs         = self.logsoftmax(class_scores)
      return logprobs

class FasterDeepRNNClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, RNNlayers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding  = nn.Embedding(len(vocab), embed_dim)  
        self.rnn_stack  = nn.RNN(embed_dim,                   #
                                 hidden_dim,
                                 RNNlayers)  
        self.linear     = nn.Linear(hidden_dim, 2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, sentence_tokens):
      all_embeddings         = self.embedding(sentence_tokens)
      all_embeddings         = all_embeddings.unsqueeze(1)
      hidden_state_history, _= self.rnn_stack(all_embeddings)

      feature_extractor_output = hidden_state_history[-1,0,:]
      class_scores     = self.linear(feature_extractor_output)
      logprobs         = self.logsoftmax(class_scores)
      return logprobs

class LSTMclassifier(FasterDeepRNNClassifier):
    def __init__(self, embed_dim, hidden_dim, RNNlayers):
      super().__init__(embed_dim, hidden_dim, RNNlayers)
      self.rnn_stack = nn.LSTM(embed_dim, hidden_dim, RNNlayers)

model     = DeepRNNClassifier(10,5,2)
optimizer = torch.optim.AdamW(model.parameters())

def print_rnn_structure(rnn_cell):
    print(f"input_linear.weight.shape: {rnn_cell.input_linear.weight.shape}")
    print(f"input_linear.bias.shape: {rnn_cell.input_linear.bias.shape}")
    print(f"hidden_linear.weight.shape: {rnn_cell.hidden_linear.weight.shape}")
    print(f"hidden_linear.bias.shape: {rnn_cell.hidden_linear.bias.shape}")

def print_model_structure(model):
    print(f"model: {model}")
    # RNN cells
    for idx, rnn_cell in enumerate(model.rnn_list):
        print(f"model rnn_cell#{idx}: {rnn_cell}")
        print_rnn_structure(rnn_cell)
    # Linear
    print(f"model.linear: {model.linear}")
    print(f"linear.weight.shape: {model.linear.weight.shape}")
    print(f"linear.bias.shape: {model.linear.bias.shape}")

    print(f"model.logsoftmax: {model.logsoftmax} - 'LogSoftmax' object has no attribute 'weight'")


print_model_structure(model)

def iterate_one_sentence(tokens, label, train_flag):
  if train_flag:
    model.train()  
    optimizer.zero_grad()
    y_model = model(tokens)
    loss    = -y_model[label] #Cross Entropy
    loss.backward()
    optimizer.step()
  else:
    model.eval()
    y_model = model(tokens)
    model.train()
  with torch.no_grad():
    predicted_labels = y_model.argmax()
    success = (predicted_labels == label)
  return success

#overfit a small batch to check if learning _can_ occur
num_samples, epochs = 100, 10
parameters = list(model.parameters())
avg_grad_norms = torch.zeros(epochs)
for epoch in range(epochs):
  correct_predictions = torch.tensor([0.])
  grad_norm_temp      = torch.zeros(num_samples)
  for idx in tqdm(range(num_samples)):
    correct_predictions += iterate_one_sentence(train_tokens[idx],
                                                train_labels[idx],
                                                train_flag=True)
    norms = [p.grad.detach().abs().max() for p in parameters if p.grad is not None]
    grad_norm_temp[idx] = torch.max(torch.stack(norms))
  avg_grad_norms[epoch] = grad_norm_temp.mean()
    
  acc = correct_predictions/num_samples
  #if epoch % 3 == 0:
  print("Epoch",epoch," acc:",acc.item())

plt.plot(avg_grad_norms)
plt.xlabel("Epoch")
plt.title("Average Inf Norm");

plt.show()

test_correct_predictions = torch.tensor([0.])
for tokens, label in tqdm(zip(test_tokens, test_labels), total=len(test_tokens)):
  test_correct_predictions += iterate_one_sentence(tokens, label, train_flag=False)
test_acc = test_correct_predictions / len(test_tokens)

print(acc, test_acc, sep="\n")
print(f"acc: {acc}, test_acc: {test_acc}")

preprocess = lambda x: torch.tensor(vocab(x.split()))
example_sentences=["very good , not bad",
                   "very bad , not good"]
with torch.no_grad():                   
  for sent in example_sentences:
    print(preprocess(sent))
    print(torch.exp(model(preprocess(sent))))