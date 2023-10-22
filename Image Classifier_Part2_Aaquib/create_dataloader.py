from torchvision import datasets, transforms, models
import torch
'''
input directory
output 1 dataloaders
output 2 image_database
'''

def create_dataloader(data_dir, batch_size=40):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    data_transforms_test_valid = transforms.Compose([transforms.Resize(250),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])])
    # TODO:d the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir,transform=data_transforms_train)
    test_image_datasets = datasets.ImageFolder(test_dir,transform=data_transforms_test_valid)
    valid_image_datasets = datasets.ImageFolder(valid_dir,transform=data_transforms_test_valid)

    # TODO: Using the image sets and the trainforms, define the dataloaders
    dataloaders ={'train':torch.utils.data.DataLoader(train_image_datasets,
                                                      batch_size=batch_size,shuffle=True),
                   'test':torch.utils.data.DataLoader(test_image_datasets, 
                                                      batch_size=batch_size, shuffle=True),
                   'valid':torch.utils.data.DataLoader(valid_image_datasets, 
                                                      batch_size=batch_size, shuffle=True)}
    image_datasets ={'train':train_image_datasets,
                   'test':test_image_datasets,
                   'valid':valid_image_datasets}
    return dataloaders, image_datasets