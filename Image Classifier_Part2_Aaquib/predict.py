import argparse
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
from predict_functions import*

parser = argparse.ArgumentParser()

parser.add_argument('image_path')
parser.add_argument('checkpoint', type = str)
parser.add_argument('--top_k', default=5, type=int)
parser.add_argument('--category_names', default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint
top_k = args.top_k
class2name_filepath = args.category_names
gpu = args.gpu

def main():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu)  else 'cpu')
    checkpoint = torch.load(checkpoint_path)
#     if torch.cuda.is_available():
#         checkpoint = torch.load(filepath)
#     else:
#         checkpoint = torch.load(filepath, map_location='cpu')
    model_name = checkpoint['modelname']
    hidden = checkpoint['layers'][1] 
    model , _ , _ = create_model(model_name,hidden)
    model = load_checkpoint(model,checkpoint_path)
    model.to(device)
    torch.cuda.empty_cache()

    ps, img_class = predict(image_path,model, top_k, device)
    names = get_flower_name(img_class, class2name_filepath)
    print(f'''\n\nFLOWER DETAILS: Class: {image_path.split('/')[-2]}    Flower Name:{get_flower_name(image_path.split('/')[-2], class2name_filepath)[0].upper()}\n''')
    print(f'''PREDICTIONS: The {top_k} most likely Classes and their Probabilities are \n''')
    for i in range(top_k):
        print(f'{i}: Class:  {img_class[i]: >3}    Probability:  {ps[i]: .3f}     Flower Name:{names[i].title()}')
    
if __name__ == "__main__":
    main()
    
# python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/checkpoint_resnet18_ep1.pth' --top_k 3 --category_names cat_to_name.json --gpu