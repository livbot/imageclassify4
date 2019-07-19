import os
import argparse
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import seaborn as sns
import time
import copy


from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
from torch import optim
from torch.optim import lr_scheduler

#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')


#%config InlineBackend.figure_format = 'retina'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

def args_parser():
    paser = argparse.ArgumentParser(description='train file')
    
    paser.add_argument('--data_directory', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False:cpu')
    paser.add_argument('--epochs', type=int, default=1, help='Number of Epochs')
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='checkpoint saved location')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate parameter')
    args = paser.parse_args()
    print('Log: Arguments value', args)
    
    return args



def check_if_gpu_available():
    gpu =torch.cuda.is_available()
    if gpu == True:
        device =  torch.device("cuda:0")
        print(device)
    else:
        device = torch.device("cpu")
        print('Log: Machine is using',device)
        
    return device
    
def process_data(train_dir, test_dir, valid_dir ):
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224), 
                                          transforms.ToTensor(), 
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True) #, num_workers =4)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True) #, num_workers=4)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True) #,num_workers=4 )
    return trainloaders, testloaders, validloaders, train_datasets, test_datasets, valid_datasets
    
def json():
    #Description: import json need to put here as it will complain the AttributeError: 'str' object
    #has no attribute load
    #To fix this, you can use another variable once loaded:
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        #print('Log: ', cat_to_name)
        #print('Log: len of cat to name', len(cat_to_name))
    return cat_to_name
	
def pytorch_algorithm(choose_algo):
    if choose_algo =='vgg16':
        pytorch_model = models.vgg16(pretrained = True)
        #print(model)
    if choose_algo =='densenet121':
        pytorch_model = models.densenet121(pretrained = True)
        print('Use densenet121')
    for param in pytorch_model.parameters():
        param.requires_grad = True
    
    return pytorch_model
## checkpoint_25088_4096_128_102_7epochs_SGD_lr0p01
##liv pytorch_model = pytorch_algorithm('vgg16')
## checkpoint 
##liv pytorch_model
##liv print(pytorch_model)

#print("Make use_algo outside the function",use_algo)

def gpu(pytorch_model, use_gpu):
    
    if not use_gpu:
        pytorch_model.cpu()
            
    else:
        pytorch_model.cuda()
   
def set_classifier(pytorch_model):
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(), 
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 102),
                                 nn.LogSoftmax(dim=1))
    return classifier

###
##liv pytorch_model.classifier = set_classifier(pytorch_model)
##liv print(pytorch_model.classifier)


#Create the network, define criterion and optimizer
#criterion = nn.CrossEntropyLoss()
def train_model(pytorch_model, epochs, device, trainloaders, testloaders, validloaders, lr):
    criterion = nn.NLLLoss()

#optimizer = optim.SGD(model.parameters(),lr=0.01)    
    optimizer = optim.SGD(pytorch_model.classifier.parameters(),lr=lr)    

    pytorch_model.to(device)   
    # Train the network
    print('Log: Train the network')
    epochs = epochs #7#5#2
    steps = 0
    print_every = 10   
    running_loss = 0

    for e in range(epochs):
        print('epochs', epochs)
        pytorch_model.train()
           
        for images, labels in trainloaders:
            steps+=1
            
            images, labels = images.to(device), labels.to(device)
            #images = images.view(images.shape[0], -1)
             
            # Clear the gradients, do this they are accumulated
            optimizer.zero_grad()
        
            #print('here first images shape',inputs.shape)
        
            logps = pytorch_model.forward(images)
            #print(logps) # print 4 images with 102 
        
            loss = criterion(logps, labels)
        
        #optimizer.zero_grad()
        #print('Before backward pass: \n', loss)
        
        #print('Gradient of Tensor \n', loss.grad)
       
            loss.backward()
        #print('After backward pass: \n', loss)
       
            optimizer.step()
        
            running_loss += loss.item()
        
            # TODO: Do validation on the test set
            #only print every 5 
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                print('  Model back to eval stage')
                pytorch_model.eval() #Turn off dropout
                with torch.no_grad():
                    for images, labels in validloaders:
                        images, labels = images.to(device), labels.to(device)
                    
                        logps = pytorch_model.forward(images)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss+= batch_loss.item()
                    
                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1) 
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                print(f"  Epoch {e+1}/{epochs}.."
                      f"  Train loss: {running_loss/print_every:.3f}.."
                      f"  Test loss: {test_loss/len(validloaders):.3f}.."
                      f"  Test accuracy: {accuracy/len(validloaders):.3f}")
                    
                print('  Model back to train stage')
                running_loss = 0
                pytorch_model.train() #set model to train mode
    print('Log: End of Train the network')
                    
###
##liv train_model(15)


#DONE: Save the checkpoint 
def save_the_checkpoint(pytorch_model, save_dir, train_datasets):
    print('Log: Save_the_checkpoint in progress .....')
    #save mapping of classes to indices which you get from the train image datasets. 
    # Attach this to model as attributes which makes inference easier later on ---- model.class_to_idx -----
    pytorch_model.class_to_idx = train_datasets.class_to_idx
#    pytorch_model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'model': pytorch_model, 
#              'epochs': epochs,
              'state_dict': pytorch_model.state_dict(),
#              'optimizer': optimizer.state_dict(),
              'class_to_idx': pytorch_model.class_to_idx}
    
    torch.save(checkpoint, save_dir)
    print('Log: Done!')
   
def main():
    
    print('Log: Main Start')
    args = args_parser()
    gpu = args.gpu
    epochs= args.epochs
    
    lr = args.lr
    
    device = check_if_gpu_available()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'  
    test_dir = data_dir + '/test'
    
    trainloaders, testloaders, validloaders, train_datasets, test_datasets, valid_datasets = process_data(train_dir,                                                                                                                                                  valid_dir,
                                                                                                         test_dir)
    json()
    
    pytorch_model = pytorch_algorithm('vgg16')
    pytorch_model
    
    ## add back later use_gpu(pytorch_model, True)
    
    pytorch_model.classifier = set_classifier(pytorch_model)
    #trmodel = train_model(pytorch_model, epochs, device, trainloaders, testloaders, validloaders)
    train_model(pytorch_model, epochs, device, trainloaders, testloaders, validloaders, lr)
    save_the_checkpoint(pytorch_model, args.save_dir, train_datasets)
    
    print('Log: Main End')
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

#save_dir = 'July_i.pth'
#save_the_checkpoint(pytorch_model, save_dir)     