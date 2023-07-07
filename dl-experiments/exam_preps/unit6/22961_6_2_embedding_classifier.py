# -*- coding: utf-8 -*-
"""22961_6_2_embedding_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Idan-Alter/OU-22961-Deep-Learning/blob/main/22961_6_2_embedding_classifier.ipynb
"""

#pip install datasets

import torch
from torch import nn
import datasets as ds
from pprint import pprint
from tqdm import tqdm

dataset = ds.load_dataset("glue", "sst2")

sentence_list = dataset["train"]["sentence"]
labels_list   = dataset["train"]["label"]
tokenize      = lambda x: x.split()
tokenized     = list(map(tokenize, sentence_list))

from torchtext.vocab import build_vocab_from_iterator
vocab = build_vocab_from_iterator(tokenized, specials=["<UNK>"], min_freq=5)
vocab.set_default_index(1)

func = lambda x: torch.tensor(vocab(x))
integer_tokens = list(map(func, tokenized))
label_tensors  = list(map(torch.tensor, labels_list))
print(*sentence_list[1:3], sep="\n")
print(*integer_tokens[1:3], sep="\n")
print(*label_tensors[1:3], sep="\n")

test_split = len(integer_tokens) * 8//10
train_tokens = integer_tokens[:test_split]
train_labels = label_tensors[:test_split]
test_tokens  = integer_tokens[test_split:]
test_labels  = label_tensors[test_split:]

class ClassificationHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, feature_extractor_output):
        class_scores= self.linear(feature_extractor_output)
        logprobs    = self.logsoftmax(class_scores)
        return logprobs

class FeatureExtractor_1(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab),embed_dim)

    def forward(self, sentence_tokens):
        embedded    = self.embedding(sentence_tokens)
        return embedded

example_sentence=sentence_list[1]

print(example_sentence)

preprocess= lambda x: torch.tensor(vocab(x.split()))
tokens = preprocess(example_sentence)
print(tokens)

extractor = FeatureExtractor_1(2)
features = extractor(tokens)
print(features, features.size(), sep="\n")

class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim, use_bag=False):
        super().__init__()
        self.use_bag = use_bag
        if not use_bag:
            self.embedding = nn.Embedding(len(vocab), embed_dim)
        else:
            self.embedding = nn.EmbeddingBag(len(vocab), embed_dim)

    def forward(self, sentence_tokens):
        if not self.use_bag:
            embedded = self.embedding(sentence_tokens)
            feature_extractor_output = embedded.sum(dim=0)    #
        else:
            sentence_tokens = sentence_tokens.unsqueeze(0)
            feature_extractor_output = self.embedding(sentence_tokens).squeeze()
        return feature_extractor_output

extractor = FeatureExtractor(2)
features  = extractor(tokens)
print(features, features.size(), sep="\n")

class EmbedSumClassify(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.extractor  = FeatureExtractor(embed_dim)
        self.classifier = ClassificationHead(embed_dim)

    def forward(self, sentence_tokens):
        extracted_features = self.extractor(sentence_tokens)
        logprobs    = self.classifier(extracted_features)
        return logprobs

model = EmbedSumClassify(2)
print(model(tokens))

model = EmbedSumClassify(12)
print(model(tokens))

def iterate_one_sentence(tokens, label, train_flag):
  tokens = tokens
  if train_flag:
    model.train()
    optimizer.zero_grad()
    y_model = model(tokens)
    loss    = -y_model[label] #CE loss
    loss.backward()
    optimizer.step()
  else:
    model.eval()
    y_model = model(tokens)
    model.train()
  with torch.no_grad():
    predicted_labels = y_model.argmax(dim=0)
    success = (predicted_labels == label)
  return success

def train_one_epoch():
  correct_predictions = torch.tensor([0.])
  for tokens, label in tqdm(zip(train_tokens, train_labels), total=len(train_tokens)):
    correct_predictions += iterate_one_sentence(tokens,label, train_flag=True)
  acc = correct_predictions / len(train_tokens)
  print("\n", acc)
  return acc

def test_model():
  test_correct_predictions = torch.tensor([0.])
  for tokens, label in tqdm(zip(test_tokens, test_labels), total=len(test_tokens)):
    test_correct_predictions += iterate_one_sentence(tokens, label, train_flag=False)
  test_acc = test_correct_predictions / len(test_tokens)
  return test_acc

model = EmbedSumClassify(5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

acc=train_one_epoch()
test_acc=test_model()

#check on random labels
test_correct_predictions = torch.tensor([0.])
random_labels = torch.rand(len(test_tokens))<0.5
for tokens, label in tqdm(zip(test_tokens, random_labels), total=len(test_tokens)):
  test_correct_predictions += iterate_one_sentence(tokens, label, train_flag=False)
rand_acc = test_correct_predictions / len(test_tokens)

print(acc, test_acc, rand_acc, sep="\n")

preprocess = lambda x: torch.tensor(vocab(x.split()))
example_sentences=["very good , not bad",
                   "very bad , not good"]
with torch.no_grad():
  for sent in example_sentences:
    print(preprocess(sent))
    print(torch.exp(model(preprocess(sent))))