Prepare the workspace
# Before you proceed, update the PATH
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
# Restart the Kernel at this point. 
# Before you proceed, update the PATH
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
# Restart the Kernel at this point. 
# Do not execute the commands below unless you have restart the Kernel after updating the PATH. 
!python -m pip install torch==1.0.0
# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 
Developing an AI application
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.


The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# Imports here
import numpy as np
import pandas as pd
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
​
from torchvision import datasets, transforms, models
import json
from PIL import Image
from collections import OrderedDict
import random
import os
Load the data
Here you'll use torchvision to load the data (documentation). The data should be included alongside this notebook, otherwise you can download it here.

If you do not find the flowers/ dataset in the current directory, /workspace/home/aipnd-project/, you can download it using the following commands.

!wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
!unlink flowers
!mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
Data Description
The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_train = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
​
data_transforms_test_valid = transforms.Compose([transforms.Resize(250),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
​
# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir,transform=data_transforms_train)
test_image_datasets = datasets.ImageFolder(test_dir,transform=data_transforms_test_valid)
valid_image_datasets = datasets.ImageFolder(valid_dir,transform=data_transforms_test_valid)
​
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train':torch.utils.data.DataLoader(train_image_datasets, batch_size=40, shuffle=True),
               'test':torch.utils.data.DataLoader(test_image_datasets, batch_size=40, shuffle=True), 
               'valid':torch.utils.data.DataLoader(valid_image_datasets, batch_size=40, shuffle=True)}
Label mapping
You'll also need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json. It's a JSON object which you can read in with the json module. This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

import json
​
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
​
Building and training the classifier
Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from torchvision.models to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:

Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
Train the classifier layers using backpropagation using the pre-trained network to get the features
Track the loss and accuracy on the validation set to determine the best hyperparameters
We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

Note for Workspace users:
If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with ls -lh), you should reduce the size of your hidden layers and train again.

# TODO: Build and train your network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
​
model = models.mobilenet_v2(pretrained = True)
print(model.classifier)
for param in model.parameters():
    param.requires_grad = False
    
torch.cuda.empty_cache()
classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(1280, 512)),
                      ('relu', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.25)),
                      
                      ('fc3', nn.Linear(512, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
model.classifier=classifier
print(model.classifier)
​
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)
a=1 #to avoid model output
​
## MODEL TRAINING
​
dataloaders = {'train':torch.utils.data.DataLoader(train_image_datasets, batch_size=40, shuffle=True),
               'test':torch.utils.data.DataLoader(test_image_datasets, batch_size=40, shuffle=True), 
               'valid':torch.utils.data.DataLoader(valid_image_datasets, batch_size=40, shuffle=True)}
#use the checkpoint file name that you want to load
model = load_checkpoint('checkpointmobilenet_v2epoch9.pth') 
epochs = 1
step = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
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
                      f"Test Loss {test_loss/len(dataloaders['valid']):.3f}.."
                      f"Test Accuracy {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()
print('======DONE=======')
checkpoint = {'layers' : [1280,512,102],
               'classidx' : train_image_datasets.class_to_idx,
               'modelname':'mobilenet_v2',
               'state_dict':model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict}
torch.save(checkpoint,'checkpointmobilenet_v2epoch10.pth')
#only validate/test
dataloaders = {'train':torch.utils.data.DataLoader(train_image_datasets, batch_size=40, shuffle=True),
               'test':torch.utils.data.DataLoader(test_image_datasets, batch_size=40, shuffle=True), 
               'valid':torch.utils.data.DataLoader(valid_image_datasets, batch_size=40, shuffle=True)}
​
model = load_checkpoint('checkpointmobilenet_v2epoch9.pth')
data_type = 'test'
​
step = 0
print_every = 5
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad(): 
    for inputs, labels in dataloaders[data_type]:
        step +=1
        inputs,labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss = criterion(logps,labels)
        test_loss += batch_loss.item()
        #accuracy
        ps = torch.exp(logps)
        top_p,top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
​
print('======DONE=======')
print(f"Step {step} .."
                  f"Test Loss {test_loss/len(dataloaders[data_type]):.3f}.."
                  f"Test Accuracy {accuracy/len(dataloaders[data_type]):.3f}")
Save the checkpoint
Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: image_datasets['train'].class_to_idx. You can attach this to the model as an attribute which makes inference easier later on.

model.class_to_idx = image_datasets['train'].class_to_idx

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, optimizer.state_dict. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# TODO: Save the checkpoint 
checkpoint = {'layers' : [1280,512,102],
               'classidx' : train_image_datasets.class_to_idx,
               'modelname':'mobilenet_v2',
               'state_dict':model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict}
torch.save(checkpoint,'checkpointmobilenet_v2epoch3.pth')
Loading the checkpoint
At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
​
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['classidx']
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model
model = load_checkpoint('checkpointmobilenet_v2epoch9.pth')
Inference for classification
Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called predict that takes an image and a model, then returns the top 𝐾
 most likely classes along with the probabilities. It should look like

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
First you'll need to handle processing the input image such that it can be used in your network.

Image Preprocessing
You'll want to use PIL to load the image (documentation). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training.

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the thumbnail or resize methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image).

As before, the network expects the images to be normalized in a specific way. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions.

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    # TODO: Process a PIL image for use in a PyTorch model
    
    img_pil = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img = transform_image(img_pil)
    #print(f'process image output{type(img), img.shape}')
    return img
To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your process_image function works, running the output through this function should return the original image (except for the cropped out portions).

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    #print(type(image), image.shape)
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
​
path = 'flowers/test/10/image_07090.jpg'
image = process_image(path)
imshow(image)
Class Prediction
Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-𝐾
) most probable classes. You'll want to calculate the class probabilities then find the 𝐾
 largest values.

To get the top 𝐾
 largest values in a tensor use x.topk(k). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using class_to_idx which hopefully you added to the model or from an ImageFolder you used to load the data (see here). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    with torch.no_grad():
        image = process_image(image_path)
        image = image.unsqueeze(0)
        
        #print(f'predict image input{type(image), image.shape}')
​
        logps = model(image)
        #print(f'sum of all probs = {torch.exp(logps).sum()}')
        img_p,idx = (torch.exp(logps)).topk(topk)
        idx = idx.numpy().squeeze()
        
        #getting class from idx
        idx_to_class = {i:c for c,i in model.class_to_idx.items()}
​
        img_class = [idx_to_class[j] for j in idx]
        img_p = img_p.numpy().squeeze()
    return img_p,img_class
def get_flower_name(img_class):
​
# Load json file
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        #print(cat_to_name)
        flower_names = []
​
        for img in img_class:
            flower_names.append(cat_to_name[str(img)])
​
    return flower_names
path = 'flowers/test/82/image_01609.jpg'
ps, img_class = predict(path,model)
names = get_flower_name(img_class)
print(ps)
print(img_class)
print(names)
Sanity Checking
Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:


You can convert from the class integer encoding to actual flower names with the cat_to_name.json file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the imshow function defined above.

def random_image(path):
    
    # Select random file
    folder = random.choice(os.listdir(path))
    
    file = random.choice(os.listdir(os.path.join(path, folder)))

    img_path = os.path.join(path, folder, file)

    return img_path
def random_image(path):
    
    # Select random file
    folder = random.choice(os.listdir(path))
    
    file = random.choice(os.listdir(os.path.join(path, folder)))
​
    img_path = os.path.join(path, folder, file)
​
    return img_path
# TODO: Display an image along with the top 5 
def sanity_check(img_path=None):
    ''' Function for viewing an image and it's predicted classes.
    '''
    if img_path == None:
        img_path = random_image('flowers/valid')
        img_folder = img_path.split('/')[-2]
    img_prob, img_class = predict(img_path, model)
    fl_names = get_flower_name(img_class)
    title = get_flower_name(img_folder.split())
    print(title[0],img_class[0], img_folder)
    
    fig = plt.figure(figsize =[8,8])
    plot1 = plt.subplot2grid((16,10), (0,0), colspan = 10, rowspan = 10)
    plot2 = plt.subplot2grid((12,10), (8,2), colspan = 6, rowspan = 5)
    
    img = Image.open(img_path)
    plot1.axis('off')
    plot1.set_title(title[0])
    plot1.imshow(img)
    
    y_ticks = np.flip(np.arange(5))
    
    plot2.set_yticks(y_ticks)
    plot2.set_yticklabels(fl_names)
    plot2.barh(y_ticks, img_prob, align='center')
    plt.show
    
 
    return
sanity_check()
cat_to_name;