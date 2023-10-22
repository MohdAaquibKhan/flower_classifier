import argparse
from get_train_input_args import get_train_input_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import json
from PIL import Image
from collections import OrderedDict
import random
import os
from create_dataloader import create_dataloader
from create_model import create_model, load_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', default="flowers")
parser.add_argument('--save_dir', default="")
parser.add_argument('--arch', default="densenet121")
parser.add_argument('--learning_rate',type = float, default=0.001)
parser.add_argument('--hidden_units', type = int, default=512)
parser.add_argument('--epochs',type = int, default=1)
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
parser.add_argument('--load_checkpoint_path', default="", help='enter file address as argument')
args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
hidden = args.hidden_units
model_name = args.arch
epochs = args.epochs
gpu = args.gpu
checkpoint_path = args.load_checkpoint_path

def main():
    torch.cuda.empty_cache()
#     print(dataloaders,'\n',gpu ,'success======')
#     with open('cat_to_name.json', 'r') as f:
#         cat_to_name = json.load(f)
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu)  else 'cpu')
#     print(device)
# Define MODEL
    model , criterion, optimizer = create_model(model_name,hidden, lr)
    if checkpoint_path:
        model = load_checkpoint(model,checkpoint_path)
    model.to(device)
    torch.cuda.empty_cache()
#     ===============================
    train(device, model , criterion, optimizer)
    
    
def train(device, model , criterion, optimizer):    
    print(f'''=========================
    Starting Training on {device}
    using model {model_name}
    with lr = {lr}
    for epochs = {epochs}
=========================''')
    dataloaders, image_datasets = create_dataloader(data_dir, batch_size = 40)
    step = 0
    running_loss = 0
    print_every = 5
#     print(model.fc)
#     print(model.classifier)

    for epoch in range(int(epochs)):
        for inputs, labels in dataloaders['train']:
            step +=1
            inputs,labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            if step%print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad(): 
                    for inputs, labels in dataloaders['valid']:
                        inputs,labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps,labels)
                        test_loss += batch_loss.item()
                        #accuracy
                        ps = torch.exp(logps)
                        top_p,top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Step {step} Epoch {epoch+1}/{epochs}.."
                              f"Train Loss {running_loss/print_every:.3f}.."
                              f"Validation Loss {test_loss/len(dataloaders['valid']):.3f}.."
                              f"Validation Accuracy {accuracy/len(dataloaders['valid']):.3f}")
                    running_loss = 0
                    model.train()
            torch.cuda.empty_cache()
    print('====== TRAINING DONE=======')
# SAVE CHECKPOINT    
    if save_dir:
        dataloaders, image_datasets = create_dataloader(data_dir, batch_size = 40)
        if model_name == 'resnet18':
            input_layer = 512
        else: input_layer = 1024
        checkpoint = {'layers' : [input_layer,hidden,102],
                      'classidx' : image_datasets['train'].class_to_idx,
                      'modelname':model_name,
                      'state_dict':model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict}
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f'checkpoint_{model_name}_ep{epochs}.pth'
        checkpoint_file_path = os.path.join(save_dir, filename)
        print(f'Checkpoint Created, File location: {checkpoint_file_path}')
        torch.save(checkpoint, checkpoint_file_path)
if __name__ == "__main__":
    main()
# python train.py 'flowers' --gpu --save_dir checkpoints --learning_rate 2 --hidden_units 128 --load_checkpoint_path 'checkpoints/checkpoint_densenet121_ep5.pth'
# python train.py 'flowers' --gpu --save_dir checkpoints --load_checkpoint_path 'checkpoints/checkpoint_resnet18_ep1.pth' --arch 'resnet18'