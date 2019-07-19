import os
import argparse
#os.environ['QT_QPA_PLATFORM']='offscreen'
#os.environ["MPLBACKEND"]="qt4agg"
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() 
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
#plt.ion()

#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

#%config InlineBackend.figure_format = 'retina'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_the_checkpoint(filepath):
    print('Log: Load_the_checkpoint..')
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    #1
    checkpoint = torch.load(filepath, map_location=map_location)
        
    pytorch_model = models.vgg16(pretrained = True)
                    
    pytorch_model.class_to_idx = checkpoint['class_to_idx']
       
    pytorch_model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 102),
                                 nn.LogSoftmax(dim=1))    
    
    for param in pytorch_model.parameters():
        param.requires_grad = False
    pytorch_model.load_state_dict(checkpoint['state_dict'])
    
        
    #for param_tensor in pytorch_model.state_dict():
    #    print('****************')
    #    print('Inside loop')
    #    print("Valid in model state_dict**", param_tensor, "\t", pytorch_model.state_dict()[param_tensor].size())   
    
    return pytorch_model

def args_parser():
    paser = argparse.ArgumentParser(description='predict file')
    
    paser.add_argument('--image_path', type=str, default='flowers/train/13/image_05744.jpg', help='Image used for Prediction')
    paser.add_argument('--checkpoint', type=str, default='July_h.pth', help='Checkpoint in *.pth')
    paser.add_argument('--top_k', type=int, default=5)
    paser.add_argument('--gpu', type=str, default=False, help='True: gpu, False:cpu')
    args = paser.parse_args()
    print('Log: Arguments value', args)
    
    return args

def gpu(pytorch_model, use_gpu):
    
    if not use_gpu:
        pytorch_model.cpu()
            
    else:
        pytorch_model.cuda()
            
#checkpoint, epochs, state_dict, pytorch_model, optimizer, class_to_idx = load_the_checkpoint('checkpoint_25088_4096_128_102_7epochs_SGD_lr0p01.pth')
#checkpoint, state_dict, pytorch_model, class_to_idx = load_the_checkpoint('July.pth')
#liv pytorch_model = load_the_checkpoint('July_h.pth')

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

import os, sys

from PIL import Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
   
    print('Log: Process image... Start!')
   
    # TODO: Process a PIL image for use in a PyTorch model
    original_im = Image.open(image)
  #  print('im(size', original_im.size)  #510,340
#    print('Image Format: {}... Image Size: {}... Image Mode: {}...'.format(original_im.format, original_im.size, original_im.mode))
    ori_width, ori_height = original_im.size
 #   print('Original image (width:{}, height:{})'.format(ori_width, ori_height))
    
    shortest = 256
    #if the shortest is width, do this
    if ori_width < ori_height:
        new_width = 256
        new_height = ori_height
        #liv print('New box (width, height)', new_width, new_height)
    else:
        new_width = ori_width
        new_height = 256
        #liv print('New box (width, height)', new_width, new_height)
      
    im_256 = original_im.resize((new_width, new_height))
 #   print('Done resize New box (width, height)', new_width, new_height)
    re_width, re_height = im_256.size
  # print('Original shortest 256 image (width:{}, height:{})'.format(re_width, re_height))
    
    mid_width = int(new_width/2)
    mid_height = int(new_height/2)
  #  print('Get middle point of the image width, height', mid_width, mid_height)
    
    im_cropped = im_256.crop((mid_width-(224//2), mid_height-(224//2),mid_width+(224//2), mid_height+(224//2)))
   # print('im_cropped', im_cropped.size)
    np_image = np.array(im_cropped)
    np_image = np_image/255 #numpy array (224, 224, 3)
    mean = [0.485, 0.456, 0.406]
    stdev = [0.229, 0.224, 0.225]
    normalized_img = (np_image - mean)/stdev
   
    #liv print('normalized_img shape',normalized_img.shape)
    # print('Checked transpose shape',normalized_img.transpose().shape)
    normalized_img = normalized_img.transpose((2, 0, 1))
    #liv print('normalized_img shape',normalized_img.shape)
    
    print('Log: Process image... Done! *****')
    #return torch.from_numpy(normalized_img.transpose())
    #return from_numpy(normalized_img.transpose())
    return normalized_img 
#normalized_img = process_image('image_06664.jpg')


def imshow(image, ax=None, title=None):
    
    print('Log: Imshow')
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    #liv print(image.shape)
    
    image = image.transpose((1, 2, 0))
    #liv print(image.shape)
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        
    image = np.clip(image, 0, 1)
 
    
    ax.imshow(image)
    
    return ax
	
#def predict(image_path, model, topk=5):
def predict(image_path, pytorch_model, cat_to_name, topk=5):
   
    print('Log: Predict ') 
     
    pytorch_model.eval()
    #print('model', model)
    img = process_image(image_path)
     
    img = torch.from_numpy(img).type(torch.FloatTensor)
    #print('img shape',img.shape)
    #print('img',img)
    img = img.unsqueeze_(0)
    #liv print(image_path)
    #liv print('after unsqueeze ---- img shape',img.shape)
        
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #force tensor to cpu
    #pytorch_model.to('cpu')
    
    #probability = torch.exp(model.forward(img))
    probability = torch.exp(pytorch_model.forward(img))
      
    
    #liv print('probability', probability)
    #print('probability.shape', probability.shape)
        
    #print('inside loop', probability)
    # no GREG top_probs, top_labs = probability.topk(topk)
   

    #top_probs, top_labs = probability.topk(topk, dim=1)
    top_probs, top_labs = probability.topk(topk)
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    print('  Log: top_labs',top_labs)
    print('  Log: top_probs',top_probs)
    
    #print('class_to_idx.items',class_to_idx.items())
    #print('class_to_idx',class_to_idx)
    idx_to_class = {val: key for key, val in pytorch_model.class_to_idx.items()}
    #print(idx_to_class)
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
       
    return top_probs, top_labels, top_flowers
    # TODO: Implement the code to predict the class from an image file

def predict_python(image_path, pytorch_model, cat_to_name, device, top_k):
   
    print('Log: Predict on non IPython ') 
    pytorch_model.eval()
    img = process_image(image_path)
    if device == 'cuda':
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    else:
        img = torch.from_numpy(img).type(torch.FloatTensor)
    
    img = img.unsqueeze_(0)
    probability = torch.exp(pytorch_model.forward(img))
    
    top_probs, top_labs = probability.topk(top_k)
    # Added TypeError: can't convert CUDA tensor to numpy
    top_probs = top_probs.cpu()
    top_labs = top_labs.cpu()
    
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    print('  Log: top_labs',top_labs)
    print('  Log: top_probs',top_probs)
     
    idx_to_class = {val: key for key, val in pytorch_model.class_to_idx.items()}
  
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
       
    return top_probs, top_labels

# TODO: Display an image along with the top 5 classes
def plot_graph(image_path, pytorch_model, cat_to_name):
   
    print('Log: Start plot graph')
    plt.figure(figsize = (6,10))
    
    ax = plt.subplot(2,1,1)
    
    flower_num = image_path.split('/')[2]
    #print('flower_num', flower_num)
    #original
    #probs, labs, flowers = predict(image_path, pytorch_model)
    
    title_ = cat_to_name[flower_num]
    #title_ = 'Try'
    #plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);
 
    #make prediction
    print(image_path)
    #probs, labs, flowers = predict(image_path, pytorch_model, topk, cat_to_name)
    probs, labs, flowers = predict(image_path, pytorch_model, cat_to_name)
    
    #plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color =sns.color_palette()[0]);
    plt.show()
    print('Log: End plot graph')

def main():
    print('Log: Main Start')
    args = args_parser()
    
    # Variables
    image_path = args.image_path
    top_k = args.top_k
    use_gpu = args.gpu
    if args.gpu == 'True':
        device = 'cuda'
    else:
        device = 'cpu'
     
    pytorch_model = load_the_checkpoint('July_h.pth')
    gpu(pytorch_model, use_gpu)
    cat_to_name = json()
    
    top_probs, top_labels = predict_python(image_path, pytorch_model, cat_to_name, device, top_k=5)
    top_flowers = [cat_to_name[lab] for lab in top_labels]
   
    if args.top_k:
        top_k_num = args.top_k
    else:
        top_k_num = 1
    for lab in range(top_k_num):
        print("Number: {}.. ".format(lab+1))
        print("Flower Labels: {}.. ".format(top_flowers[lab]))
        print("Probability: {:.2f}..% ".format(top_probs[lab]*100))
        print('')

    #plot_graph('flowers/train/13/image_05744.jpg', pytorch_model, cat_to_name)
    
    print('Log: predict.py Main End')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)