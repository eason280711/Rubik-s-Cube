import torch
import torch.nn as nn
from torchsummary import summary

import dataset

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.fc1 = nn.Linear(60, 512)
        self.fc2 = nn.Linear(512, 786)
        self.fc3 = nn.Linear(786, 1024)
        self.fc4 = nn.Linear(1024, 1)

    def forward(self, src, tgt):
        out = torch.cat((src,tgt),1)
        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)
        out = self.fc3(out)
        out = nn.functional.relu(out)
        out = self.fc4(out)
        return out

class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        self.fc1 = nn.Linear(40, 512)
        self.fc2 = nn.Linear(512, 786)
        self.fc3 = nn.Linear(786, 1024)
        self.fc4 = nn.Linear(1024, 30)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)
        out = self.fc3(out)
        out = nn.functional.relu(out)
        out = self.fc4(out)
        return out

class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        self.fc1 = nn.Linear(30, 1024)
        self.fc2 = nn.Linear(1024, 786)
        self.fc3 = nn.Linear(786, 512)
        self.fc4 = nn.Linear(512, 40)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)
        out = self.fc3(out)
        out = nn.functional.relu(out)
        out = self.fc4(out)
        return out
        
#main
if __name__ == "__main__":
    
    # 定義超參數
    d_model = 256
    nhead = 16
    num_layers = 6
    dropout = 0.1
    seq_length = 54
    batch_size = 16
    vocab_size = 54
    
    encoder = TransformerModel(vocab_size, d_model, nhead, num_layers, dropout,seq_length).cuda()
    decoder = Decoder().cuda()
    discriminator = Discriminator().cuda()

    src = torch.randint(0, 7, (16,6,3,3)).cuda()
    tgt = src.clone().detach()

    #z_mean, z_log_var
    output = encoder(src.view(-1,54).transpose(0,1),tgt.view(-1,54).transpose(0,1))

    #output = z_mean.float() + torch.exp(0.5 * z_log_var.float()) * torch.randn_like(z_mean.float())

    d_output = decoder(output)

    # src.view(-1, 54)

    # output = encoder(src, tgt)
    # print("Output shape:", output.shape)
    
    # # generate a random (6,3,3) tensor
    # x = torch.rand((1, 6, 3, 3)).cuda()
    
    # # pass the tensor through the encoder to generate bottleneck
    # z_mean, z_log_var = encoder(x)
    # z = z_mean + torch.exp(0.5 * z_log_var) * torch.randn_like(z_mean)
    
    # # pass the bottleneck through the decoder to generate output
    # output = decoder(z)
    
    # print the input and output sizes for the encoder
    #summary(encoder,input_size=[(6,3,3),(6,3,3)])
    
    # print the input and output sizes for the decoder
    #summary(decoder, (54,))
