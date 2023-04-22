import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np

import torch.optim.lr_scheduler as lr_scheduler

import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout, seq_length,device):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.generate_positional_encoding(d_model, seq_length).to(device)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dropout=dropout)
        self.fc1 = nn.Linear(d_model, vocab_size)
        self.fc21 = nn.Linear(seq_length*vocab_size,54)
        self.fc22 = nn.Linear(54,40)
        self.device = device

    def forward(self, src, tgt):
        
        src = self.embedding(src.long()) + self.positional_encoding
        tgt = self.embedding(tgt.long()) + self.positional_encoding
        output = self.transformer(src, tgt)
        output = self.fc1(output).transpose(0,1)
        #print(output.shape)
        output = output.reshape(-1, output.shape[1]*output.shape[2])
        #print(output.shape)
        output = self.fc21(output)
        output = nn.functional.relu(output)
        output = self.fc22(output)
        return output

    def generate_positional_encoding(self, d_model, seq_length):
        pe = torch.zeros(seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 786)
        self.fc3 = nn.Linear(786, 1024)
        self.fc4 = nn.Linear(1024, 54)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)
        out = self.fc3(out)
        out = nn.functional.relu(out)
        out = self.fc4(out)
        out = out.view(-1,6,3,3)
        return out


# load model
        
batch_size = 8
lr = 0.00002
beta1 = 0.5
beta2 = 0.999
epochs = 1
total_epochs = 1

# 定義超參數
d_model = 256
nhead = 32
num_layers = 7
dropout = 0.1
seq_length = 54
vocab_size = 7
    
encoder = TransformerModel(vocab_size, d_model, nhead, num_layers, dropout, seq_length,device).to(device)

encoder.load_state_dict(torch.load("./encoder.pth"))

decoder = Decoder().cuda()
decoder.load_state_dict(torch.load("./decoder.pth"))

encoder.eval()
decoder.eval()

train_data = dataset.dataset("./datatest2",device)

criterion = nn.MSELoss()

triplet_loss = nn.TripletMarginLoss(margin=1, p=2)

idx = 0
for i in train_data:
    initial_state = i[0].cuda()
    destination_state_1 = i[1].cuda()
    destination_state_2 = i[2].cuda()
    distance_1 = i[3].cuda()
    distance_2 = i[4].cuda()

    # Encoding
    src = initial_state.view(-1,54).transpose(0,1)
    tgt = initial_state.view(-1,54).transpose(0,1).clone().detach()
    
    #z_mean_1, z_log_var_1/
    z_1 = encoder(src,tgt)
    #z_1 = z_mean_1.float() + torch.exp(0.5 * z_log_var_1.float()) * torch.randn_like(z_mean_1.float())
    initial_state_bottleneck = z_1

    src_2 = destination_state_1.view(-1,54).transpose(0,1)
    tgt_2 = destination_state_1.view(-1,54).transpose(0,1).clone().detach()
    
    #z_mean_2, z_log_var_2/
    z_2 = encoder(src_2,tgt_2)
    #z_2 = z_mean_2.float() + torch.exp(0.5 * z_log_var_2.float()) * torch.randn_like(z_mean_2.float())
    destination_state_1_bottleneck = z_2

    src_3 = destination_state_2.view(-1,54).transpose(0,1)
    tgt_3 = destination_state_2.view(-1,54).transpose(0,1).clone().detach()

    #z_mean_3, z_log_var_3/
    z_3 = encoder(src_3,tgt_3)
    #z_3 = z_mean_3.float() + torch.exp(0.5 * z_log_var_3.float()) * torch.randn_like(z_mean_3.float())
    destination_state_2_bottleneck = z_3

    # Decoding
    initial_state_reconstructed = decoder(initial_state_bottleneck)
    destination_state_1_reconstructed = decoder(destination_state_1_bottleneck)
    destination_state_2_reconstructed = decoder(destination_state_2_bottleneck)

    reconstruction_loss = criterion(initial_state_reconstructed, initial_state) + criterion(destination_state_1_reconstructed, destination_state_1) + criterion(destination_state_2_reconstructed, destination_state_2)

    triplet_margin_loss = triplet_loss(initial_state_bottleneck, destination_state_2_bottleneck, destination_state_1_bottleneck)

    print("index: ", idx)
    print("reconstruction loss: ", reconstruction_loss)#penis
    print("triplet margin loss: ", triplet_margin_loss)
    print("--------------------")

    idx += 1