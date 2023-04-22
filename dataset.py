import os
import json

import torch
import torch.utils.data as data

class dataset(data.Dataset):
    def __init__(self,img_dir,device):
        self.img_dir = img_dir
        self.list_files = os.listdir(self.img_dir)
        self.colormap = {'U': 1,'R': 2,'F': 3,'D': 4,'L': 5,'B': 6}
        self.device = device
    
    def __len__(self):
        return len(self.list_files)

    def map_data(self,data):
        mapped_array = []

        for inner_list in data:
            mapped_inner_list = []
            for sub_inner_list in inner_list:
                mapped_sub_inner_list = [self.colormap[char] for char in sub_inner_list]
                mapped_inner_list.append(mapped_sub_inner_list)
            mapped_array.append(mapped_inner_list)

        return mapped_array
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.img_dir,img_file)

        with open(img_path, 'r') as f:
            data = json.load(f)
            
            data['initial_state'] = self.map_data(data['initial_state'])
            data['destination_state_1'] = self.map_data(data['destination_state_1'])
            data['destination_state_2'] = self.map_data(data['destination_state_2'])

            #print(data)
        return torch.FloatTensor(data['initial_state']).to(self.device), torch.FloatTensor(data['destination_state_1']).to(self.device),torch.FloatTensor(data['destination_state_2']).to(self.device), torch.FloatTensor([data['distance_1']]).to(self.device), torch.FloatTensor([data['distance_2']]).to(self.device)