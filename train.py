import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np

import torch.optim.lr_scheduler as lr_scheduler

import dataset
import model

import os

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/AE")

torch.autograd.set_detect_anomaly(True)

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
    
encoder = model.TransformerModel(vocab_size, d_model, nhead, num_layers, dropout, seq_length,device).to(device)
decoder = model.Decoder().to(device)
discriminator = model.Discriminator().to(device)

encoder2 = model.Encoder2().to(device)

encoder.load_state_dict(torch.load("./encoder.pth"))

decoder2 = model.Decoder2().to(device)

decoder.load_state_dict(torch.load("./decoder.pth"))

encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, beta2))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

encoder_scheduler = lr_scheduler.LambdaLR(encoder_optimizer, lr_lambda=lambda epoch: 1.0 - epoch / total_epochs)
decoder_scheduler = lr_scheduler.LambdaLR(decoder_optimizer, lr_lambda=lambda epoch: 1.0 - epoch / total_epochs)
discriminator_scheduler = lr_scheduler.LambdaLR(discriminator_optimizer, lr_lambda=lambda epoch: 1.0 - epoch / total_epochs)

criterion = nn.MSELoss()
triplet_loss = nn.TripletMarginLoss(margin=1, p=2)

train_data = dataset.dataset("./dataset2",device)
print(device)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, (initial_state, destination_state_1 , destination_state_2 , distance_1, distance_2) in enumerate(train_loader):
        #------------------------------------------#

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()

        #------------------------------------------#

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
        
        # test #################################

        # test_init = encoder2(initial_state_bottleneck)
        # test_dest_1 = encoder2(destination_state_1_bottleneck)
        # test_dest_2 = encoder2(destination_state_2_bottleneck)

        # res_init = decoder2(test_init)
        # res_dest_1 = decoder2(test_dest_1)
        # res_dest_2 = decoder2(test_dest_2)

        # test1_loss = criterion(initial_state_bottleneck, res_init) + criterion(destination_state_1_bottleneck, res_dest_1) + criterion(destination_state_2_bottleneck, res_dest_2)

        # discrimination_1_output = discriminator(test_init,test_dest_1)
        # discrimination_2_output = discriminator(test_init,test_dest_2)

        # test2_loss = criterion(discrimination_1_output, distance_1) + criterion(discrimination_2_output, distance_2)

        # test_loss = test1_loss + test2_loss

        ########################################

        # Decoding
        initial_state_reconstructed = decoder(initial_state_bottleneck)
        destination_state_1_reconstructed = decoder(destination_state_1_bottleneck)
        destination_state_2_reconstructed = decoder(destination_state_2_bottleneck)

        reconstruction_loss = criterion(initial_state_reconstructed, initial_state) + criterion(destination_state_1_reconstructed, destination_state_1) + criterion(destination_state_2_reconstructed, destination_state_2)
        
        distance_loss = triplet_loss(initial_state_bottleneck,destination_state_2_bottleneck,destination_state_1_bottleneck)
        ########## Discrimination ##########
        
        # discrimination_1_output = discriminator(initial_state_bottleneck,destination_state_1_bottleneck)

        # discrimination_2_output = discriminator(initial_state_bottleneck,destination_state_2_bottleneck)
        
        # discriminator_loss = criterion(discrimination_1_output, distance_1) + criterion(discrimination_2_output, distance_2)

        #triplet_margin_loss = triplet_loss(initial_state_bottleneck, destination_state_2_bottleneck, destination_state_1_bottleneck)
        
        # calculate distance

        # distance_fake = torch.sum((initial_state_bottleneck - destination_state_bottleneck) ** 2)

        # distance = distance ** 2

        # distance_loss = criterion(distance_fake, distance)

        # Combine the losses
        combined_loss = 0.08 * reconstruction_loss + distance_loss
        combined_loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        discriminator_optimizer.step()
        #------------------------------------------#

        # Discrimination

        #------------------------------------------#

        # encoder.train()

        # initial_bottleneck_2 = encoder(initial_state)
        # destination_bottleneck_2 = encoder(destination_state)

        # concatenated_bottlenecks_2 = torch.cat((initial_bottleneck_2,destination_bottleneck_2), 1)
        # discrimination_output_2 = discriminator(concatenated_bottlenecks_2)

        # encoder_discriminator_loss = -1 * criterion(discrimination_output_2, distance)

        # encoder_discriminator_loss.backward()
        # encoder_optimizer.step()
        
        #------------------------------------------#
        
        # Optimization
        
        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        # discriminator_optimizer.zero_grad()

        #------------------------------------------#

        writer.add_scalar('Reconstruction Loss', reconstruction_loss, epoch * len(train_loader) + i)
        writer.add_scalar('Distance Loss', distance_loss, epoch * len(train_loader) + i)
        writer.add_scalar('Combined Loss', combined_loss, epoch * len(train_loader) + i)


        if i % 10 == 0:
            print("Epoch: {}/{} | Batch: {}/{} | Reconstruction loss: {} | Distance loss: {} | Total loss: {}".format(epoch, epochs, i, len(train_loader), reconstruction_loss, distance_loss, combined_loss))

torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

writer.close()
