
import argparse
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import seaborn as sns

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Training')

    parser.add_argument('--data_dir')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=150)
    parser.add_argument('--output_features', help='Specify the number of output features', type=int, default=102)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=6)
    parser.add_argument('--gpu', help='Use GPU for training', default='cpu')
    parser.add_argument('--arch', help='Choose architecture', default='resnet152')

    return parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
     
# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=80)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=35)

def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())

    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units, output_features):
    if hasattr('model', 'classifier'):
        in_features = model.classifier.in_features
    else:
        in_features = model.fc.in_features

    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(hidden_units, output_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=6, print_every=10, step=0):
    for epoch in range(epochs):
        running_loss = 0

        for images, labels in trainloader:
            step += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0:
                test_loss = 0
                correct = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        _, top_class_idx = ps.topk(1, dim=1)
                        equals = top_class_idx == labels.view(*top_class_idx.shape)
                        correct += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {correct/len(validloader):.3f}")
                running_loss = 0
                model.train()

    return model

def save_checkpoint(model, optimizer, class_to_idx, path, arch, hidden_units, output_features):

    model.class_to_idx = class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'arch': arch,
                  'hidden_units': hidden_units,
                  'output_features': output_features
                  }
    torch.save(checkpoint, path)

def main():
    args = arg_parse()

    data_dir = args.data_dir
    save_path = args.save_dir
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_features = args.output_features
    epochs = args.epochs
    gpu = args.gpu
    arch = args.arch

    if args.gpu:
        device = check_device()

    model = load_model(arch)

    if hasattr('model', 'classifier'):
        model.classifier = initialize_classifier(model, hidden_units, output_features)
    else:
        model.fc = initialize_classifier(model, hidden_units, output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model.to(device)

    print_every = 10
    steps = 0

    train_model(model, trainloader, validloader, device, optimizer, criterion, epochs, print_every, steps)
    save_checkpoint(model, optimizer, train_data.class_to_idx, save_path, arch, hidden_units, output_features)

if __name__ == '__main__':
    main()