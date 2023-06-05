import torch
from torch import nn
import datasets as ds  # install from https://github.com/huggingface/datasets
from pprint import pprint
from torchtext.vocab import build_vocab_from_iterator


class ClassificationHead(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, feature_extractor_output):
        class_scores= self.linear(feature_extractor_output)
        logprobs    = self.logsoftmax(class_scores)
        return logprobs


class FeatureExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim)

    def forward(self, sentence_tokens):
        embedded    = self.embedding(sentence_tokens)
        feature_extractor_output = embedded.sum(dim=0)    #
        return feature_extractor_output


class EmbedSumClassify(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.extractor  = FeatureExtractor(embed_dim)
        self.classifier = ClassificationHead(embed_dim)

    def forward(self, sentence_tokens):
        extracted_features = self.extractor(sentence_tokens)
        logprobs    = self.classifier(extracted_features)
        return logprobs

if __name__ == '__main__':
    print("maman 16 ex 1")
    dataset = ds.load_dataset("glue", "sst2")
    pprint(dataset)
    pprint(dataset["train"])
    pprint(dataset["train"][1:3])

    sentence_list = dataset["train"]["sentence"]
    labels_list = dataset["train"]["label"]
    print(sentence_list[1], labels_list[1], sep="\n")

    sentence = dataset["train"]["sentence"][1]
    print(sentence, type(sentence))
    print(sentence.split())

    tokenizer = lambda x: x.split()
    tokenized = list(map(tokenizer, sentence_list))
    print(f"len sentence_list: {len(sentence_list)}")
    print(f"len tokenized: {len(tokenized)}")

    vocab = build_vocab_from_iterator(tokenized)
    print(vocab(sentence.split()))
    print(vocab("one two three".split()))

    print(vocab.get_itos()[0:10])

    pprint(f"labels_list: {labels_list[1:3]}")
    pprint(sentence_list[1:3])
    print(*tokenized[1:3], sep="\n")

    vocab = build_vocab_from_iterator(tokenized, specials=["<UNK>"], min_freq=5)
    vocab.set_default_index(0)
    vocab("hello world".split())

    stoi = lambda x: torch.tensor(vocab(x))  # string to integer
    integer_tokens = list(map(stoi, tokenized))
    print(integer_tokens[1])

    # Embedding
    embedding_layer = nn.Embedding(len(vocab), 3)
    integer = stoi(["word"])
    print(integer)
    print(embedding_layer(integer))

    # --------------------------------

    example_sentence = sentence_list[1]
    preprocess = lambda x: torch.tensor(vocab(x.split()))
    tokens = preprocess(example_sentence)
    print(tokens)

    model = EmbedSumClassify(2)
    print(model(tokens))

