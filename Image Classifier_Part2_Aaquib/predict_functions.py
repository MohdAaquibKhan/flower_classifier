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

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image_path)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img = transform_image(img_pil)
    #print(f'process image output{type(img), img.shape}')
    return img

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.cpu()
    with torch.no_grad():
        image = process_image(image_path)
        image = image.unsqueeze(0)
        
        #print(f'predict image input{type(image), image.shape}')

        logps = model(image)
        #print(f'sum of all probs = {torch.exp(logps).sum()}')
        img_p,idx = (torch.exp(logps)).topk(topk)
        idx = idx.numpy().squeeze()
        
        #getting class from idx
        idx_to_class = {i:c for c,i in model.class_to_idx.items()}

        img_class = [idx_to_class[j] for j in idx]
        img_p = img_p.numpy().squeeze()
    return img_p,img_class

def get_flower_name(img_class, class2name_filepath = 'cat_to_name.json' ):

    with open(class2name_filepath, 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
        flower_names = []

        for img in img_class:
            flower_names.append(cat_to_name[str(img)])

    return flower_names