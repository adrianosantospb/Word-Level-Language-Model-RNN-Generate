import torch
from src.core.helpers import get_text_chunks, get_text_encoded, get_text_from, get_vocabulary, get_word_to_id, save_weights, tokenize
from src.datasets.fairytales import FairyTalesDataset
from src.models.rnn import RNN
from src.train import training
from src.test import generate
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse

# Cleaning CUDA
torch.cuda.empty_cache()

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)

# dataset file path
file_path = "./src/datasets/dataset.txt"

# get text from dataset
text = get_text_from(file_path)

# get tokens
tokens = tokenize(text)

# get vocabulary from tokens
vocabulary = get_vocabulary(tokens)

# get word_id by word
word2id = get_word_to_id(vocabulary)

# get text encoded
text_encoded = get_text_encoded(tokens, word2id)

# text chunked
text_chunks = get_text_chunks(text_encoded)

# Dataset
batch_size = 64
num_worker = 4 * int(torch.cuda.device_count())
train_dataset = FairyTalesDataset(text_chunks)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True)

# hyper-parameters
voc_size = len(vocabulary)
embed_dim = 256
rnn_hidden_size = 512
seq_length = 50

# model
model = RNN(voc_size, embed_dim, rnn_hidden_size)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
l_r = 0.005
optim = torch.optim.Adam(parameters, lr=l_r)
n_epochs = 100
best_loss = 100
dir_base = "./weigths"

def train(model, criterion, optim, train_dataloader, seq_length, device):
    # Training loop
    for epoch in range(1, n_epochs):
        
        print(' \n *********** Epoch {} *********** \n'.format(epoch))
        _, _, _, loss = training(model, criterion, optim, train_dataloader, seq_length, device)
        
        loss = loss.item()/seq_length
        
        print(f'Epoch {epoch} loss: {loss:.4f}')
        if loss < best_loss:
            save_weights(model, dir_base, epoch, best_loss)
            best_loss = loss

    # Cleaning CUDA
    torch.cuda.empty_cache()

def test(model, tokenize, word2id, word_array, seed_str, len_generated_text=50, temperature=1, top_p=0.95, device=device):
    return generate(model, tokenize, word2id, word_array, seed_str, len_generated_text, temperature, top_p, device=device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', default="test", choices=["train", "test"])
    parser.add_argument('-s', default="patience")

    parser = parser.parse_args()
    
    if parser.a == "train":
        train(model, criterion, optim, train_dataloader, seq_length, device)
    else:
        seed = parser.s
        model_test = RNN(voc_size, embed_dim, rnn_hidden_size)
        model_test = model.to(device)
        model_test.load_state_dict(torch.load(dir_base + "/best.pt")['model'])
        model_test.eval()
        with torch.no_grad():
            output = test(model_test, tokenize, word2id, np.array(vocabulary), seed, len_generated_text=60, temperature=1, top_p=0.95, device=device)
            print(output)