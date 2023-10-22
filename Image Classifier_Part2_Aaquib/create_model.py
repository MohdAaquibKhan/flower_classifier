import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from collections import OrderedDict

def create_model(model_name, hidden, lr=0.001):    
    if model_name == 'resnet18':
        input_layers = 512
        model = models.resnet18(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    
        torch.cuda.empty_cache()
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layers, hidden)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.25)),

                              ('fc3', nn.Linear(hidden, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model.fc=classifier
#         print(model.fc)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        input_layers = 1024
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    
        torch.cuda.empty_cache()
        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_layers, hidden)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(p=0.25)),

                              ('fc3', nn.Linear(hidden, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
        model.classifier=classifier
#         print(model.classifier)

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    return model , criterion, optimizer
def load_checkpoint(model, filepath):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['classidx']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model